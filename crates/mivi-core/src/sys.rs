//! System diagnostics and process telemetry utilities.

/// Fallback RSS estimate when /proc/self/statm is unavailable.
const FALLBACK_RSS_MB: f32 = 128.0;
/// Linux procfs statm path.
#[cfg(target_os = "linux")]
const PROCMEM_STATM_PATH: &str = "/proc/self/statm";
/// Linux sysconf identifier for _SC_PAGESIZE.
#[cfg(target_os = "linux")]
const SC_PAGESIZE: i32 = 30;
/// Default OS page size when sysconf fails or on non-Linux platforms.
const DEFAULT_PAGE_SIZE: usize = 4096;

/// Estimate the resident set size (RSS) memory of the current process in megabytes (MB).
pub fn estimate_process_memory_mb() -> f32 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string(PROCMEM_STATM_PATH) {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<usize>() {
                    let page_size_bytes = get_system_page_size();
                    return (pages as f64 * page_size_bytes as f64 / (1024.0 * 1024.0)) as f32;
                }
            }
        }
    }
    FALLBACK_RSS_MB
}

/// Retrieve the OS memory page size in bytes.
#[cfg(target_os = "linux")]
pub fn get_system_page_size() -> usize {
    extern "C" {
        fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    }
    // SAFETY: sysconf is a POSIX libc function taking valid integer parameter constants.
    let res = unsafe { sysconf(SC_PAGESIZE) };
    if res > 0 {
        res as usize
    } else {
        DEFAULT_PAGE_SIZE
    }
}

#[cfg(not(target_os = "linux"))]
pub fn get_system_page_size() -> usize {
    DEFAULT_PAGE_SIZE
}
