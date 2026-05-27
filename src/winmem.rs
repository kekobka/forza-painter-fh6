//! Windows live-process memory access — port of native.py (+ fh6_probe.py's
//! VirtualQueryEx region walk). Windows-only.

use std::ffi::c_void;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::Debug::{ReadProcessMemory, WriteProcessMemory};
use windows::Win32::System::Memory::{
    VirtualQueryEx, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_PRIVATE, PAGE_GUARD, PAGE_NOACCESS,
};
use windows::Win32::System::ProcessStatus::EnumProcessModules;
use windows::Win32::System::Threading::{OpenProcess, PROCESS_ACCESS_RIGHTS};

const PROCESS_ALL_ACCESS: PROCESS_ACCESS_RIGHTS = PROCESS_ACCESS_RIGHTS(0x1F0FFF);
// native.py READABLE_WRITABLE_MASK
const READABLE_WRITABLE_MASK: u32 = 0xCC;

pub struct Proc {
    handle: HANDLE,
    #[allow(dead_code)]
    pub pid: u32,
}

impl Drop for Proc {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.handle);
        }
    }
}

impl Proc {
    pub fn open(pid: u32) -> Result<Self, String> {
        let handle = unsafe { OpenProcess(PROCESS_ALL_ACCESS, false, pid) }
            .map_err(|e| format!("OpenProcess failed for pid {pid}: {e}. Run as administrator."))?;
        Ok(Self { handle, pid })
    }

    /// native.get_base_address: first enumerated module is the main module base.
    pub fn base_address(&self) -> Result<u64, String> {
        let mut modules = [windows::Win32::Foundation::HMODULE::default(); 1024];
        let mut needed = 0u32;
        unsafe {
            EnumProcessModules(
                self.handle,
                modules.as_mut_ptr(),
                std::mem::size_of_val(&modules) as u32,
                &mut needed,
            )
        }
        .map_err(|e| format!("EnumProcessModules failed: {e}"))?;
        Ok(modules[0].0 as u64)
    }

    /// native.read_process_memory: tolerate ERROR_PARTIAL_COPY, return the
    /// bytes that were actually read (same semantics the Python relies on).
    pub fn read(&self, address: u64, size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size];
        let mut read = 0usize;
        let _ = unsafe {
            ReadProcessMemory(
                self.handle,
                address as *const c_void,
                buf.as_mut_ptr() as *mut c_void,
                size,
                Some(&mut read),
            )
        };
        buf.truncate(read);
        buf
    }

    pub fn write(&self, address: u64, data: &[u8]) -> Result<(), String> {
        let mut written = 0usize;
        unsafe {
            WriteProcessMemory(
                self.handle,
                address as *const c_void,
                data.as_ptr() as *const c_void,
                data.len(),
                Some(&mut written),
            )
        }
        .map_err(|e| format!("WriteProcessMemory @0x{address:x} failed: {e}"))
    }

    pub fn read_u32(&self, address: u64) -> u32 {
        let b = self.read(address, 4);
        if b.len() == 4 {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            0
        }
    }

    pub fn read_u16(&self, address: u64) -> u16 {
        let b = self.read(address, 2);
        if b.len() == 2 {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            0
        }
    }

    pub fn read_u64(&self, address: u64) -> u64 {
        let b = self.read(address, 8);
        if b.len() == 8 {
            u64::from_le_bytes(b[..8].try_into().unwrap())
        } else {
            0
        }
    }

    /// native.dereference_pointer
    pub fn deref(&self, address: u64) -> u64 {
        self.read_u64(address)
    }

    /// native.scan_block: read a block and find a byte signature in it.
    pub fn scan_block(&self, start: u64, size: u64, needle: &[u8]) -> i64 {
        let mem = self.read(start, size as usize);
        find_sub(&mem, needle).map(|p| p as i64).unwrap_or(-1)
    }

    /// fh6_probe.iter_regions: committed readable/writable regions. When
    /// `private_only` is set, restrict to MEM_PRIVATE (the FH6 layout-count
    /// locator only scans private writable memory).
    pub fn regions(&self, private_only: bool) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut address: u64 = 0x10000;
        let max_address: u64 = 0x7FFF_FFFF_FFFF;
        while address < max_address {
            let mut info = MEMORY_BASIC_INFORMATION::default();
            let r = unsafe {
                VirtualQueryEx(
                    self.handle,
                    Some(address as *const c_void),
                    &mut info,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if r == 0 {
                address += 0x10000;
                continue;
            }
            let base = info.BaseAddress as u64;
            let size = info.RegionSize as u64;
            if size == 0 {
                address += 0x1000;
                continue;
            }
            let protect = info.Protect.0;
            let committed = info.State == MEM_COMMIT;
            let guarded = (protect & PAGE_GUARD.0) != 0 || (protect & PAGE_NOACCESS.0) != 0;
            let rw = (protect & READABLE_WRITABLE_MASK) != 0;
            let type_ok = !private_only || info.Type == MEM_PRIVATE;
            if committed && !guarded && rw && type_ok {
                out.push((base, size));
            }
            let next = base + size;
            if next <= address {
                break;
            }
            address = next;
        }
        out
    }

    /// fh6_probe.is_private_writable_address — single-address VirtualQueryEx.
    pub fn is_private_writable(&self, address: u64) -> bool {
        if !is_user_pointer(address) {
            return false;
        }
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let r = unsafe {
            VirtualQueryEx(
                self.handle,
                Some(address as *const c_void),
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if r == 0 {
            return false;
        }
        let protect = info.Protect.0;
        let guarded = (protect & PAGE_GUARD.0) != 0 || (protect & PAGE_NOACCESS.0) != 0;
        info.State == MEM_COMMIT && !guarded && (protect & READABLE_WRITABLE_MASK) != 0
    }
}

pub fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

pub fn is_user_pointer(v: u64) -> bool {
    (0x10000..=0x7FFF_FFFF_FFFF).contains(&v)
}
