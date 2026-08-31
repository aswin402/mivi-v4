//! Resource safety watchdog that monitors RAM and CPU usage to prevent system freeze/OOM.

use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Configuration for the resource safety watchdog.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// RAM usage warning threshold in MB (default: 700 MB).
    pub warn_mb: f32,
    /// RAM usage emergency shutdown threshold in MB (default: 900 MB).
    pub kill_mb: f32,
    /// Memory check polling interval (default: 3 seconds).
    pub check_interval: Duration,
    /// Whether the watchdog is enabled.
    pub enabled: bool,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            warn_mb: 700.0,
            kill_mb: 900.0,
            check_interval: Duration::from_secs(3),
            enabled: true,
        }
    }
}

/// Resource watchdog supervisor task.
pub struct ResourceWatchdog {
    config: WatchdogConfig,
    shutdown_tx: watch::Sender<bool>,
}

impl ResourceWatchdog {
    /// Create and spawn a new background watchdog task.
    ///
    /// Returns the shutdown signal receiver (`watch::Receiver<bool>`) and the background task join handle.
    pub fn spawn(config: WatchdogConfig) -> (watch::Receiver<bool>, JoinHandle<()>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let watchdog = Self {
            config,
            shutdown_tx,
        };

        let handle = tokio::spawn(async move {
            watchdog.run_loop().await;
        });

        (shutdown_rx, handle)
    }

    async fn run_loop(self) {
        if !self.config.enabled {
            return;
        }

        let mut interval = tokio::time::interval(self.config.check_interval);
        let mut warned_recently = false;

        loop {
            interval.tick().await;

            let current_rss = mivi_core::estimate_process_memory_mb();

            // Emergency kill threshold exceeded
            if current_rss >= self.config.kill_mb {
                eprintln!(
                    "\n  \x1b[1;31m🛑 [mivi safelock]\x1b[0m Emergency safety shutdown! RAM usage \x1b[1;31m{:.1} MB\x1b[0m exceeded limit \x1b[1;31m{:.0} MB\x1b[0m to protect system from lagging/freeze.",
                    current_rss, self.config.kill_mb
                );
                eprintln!("  \x1b[31mSaving state and terminating server gracefully...\x1b[0m\n");
                let _ = self.shutdown_tx.send(true);
                break;
            }

            // Warning threshold exceeded
            if current_rss >= self.config.warn_mb {
                if !warned_recently {
                    eprintln!(
                        "  \x1b[1;33m⚠ [mivi safety]\x1b[0m High RAM usage: \x1b[33m{:.1} MB\x1b[0m / {:.0} MB (threshold: {:.0} MB)",
                        current_rss, self.config.kill_mb, self.config.warn_mb
                    );
                    warned_recently = true;
                }
            } else {
                warned_recently = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_config_default() {
        let config = WatchdogConfig::default();
        assert_eq!(config.warn_mb, 700.0);
        assert_eq!(config.kill_mb, 900.0);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_watchdog_spawn_and_shutdown_trigger() {
        let config = WatchdogConfig {
            warn_mb: 1.0,
            kill_mb: 1.0, // Instantly exceeds RSS threshold
            check_interval: Duration::from_millis(20),
            enabled: true,
        };
        let (mut rx, handle) = ResourceWatchdog::spawn(config);
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(*rx.borrow_and_update());
        let _ = handle.await;
    }

    #[tokio::test]
    async fn test_watchdog_disabled_does_not_trigger() {
        let config = WatchdogConfig {
            warn_mb: 0.1,
            kill_mb: 0.1,
            check_interval: Duration::from_millis(20),
            enabled: false,
        };
        let (rx, handle) = ResourceWatchdog::spawn(config);
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(!*rx.borrow());
        let _ = handle.await;
    }
}
