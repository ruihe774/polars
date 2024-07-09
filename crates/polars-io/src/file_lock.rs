#[cfg(target_family = "unix")]
mod in_process {
    // Use in-process locking.

    use std::collections::btree_map::Entry;
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::io;
    use std::num::{NonZeroU128, NonZeroU32};
    use std::os::unix::fs::MetadataExt;
    use std::sync::Mutex;

    use polars_error::{polars_bail, PolarsResult};

    type Key = NonZeroU128;
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    enum State {
        WriteLocked,
        ReadLocked(NonZeroU32),
    }
    static LOCKED_FILES: Mutex<BTreeMap<Key, State>> = Mutex::new(BTreeMap::new());

    fn make_key(file: &File) -> io::Result<NonZeroU128> {
        let metadata = file.metadata()?;
        let dev = metadata.dev() as u128;
        let ino = metadata.ino() as u128;
        Ok(NonZeroU128::new(dev << 64 | ino).unwrap())
    }

    pub fn lock(file: &mut File, write: bool) -> PolarsResult<()> {
        if let Ok(key) = make_key(file) {
            let mut locks = LOCKED_FILES.lock().unwrap();
            match locks.entry(key) {
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    State::ReadLocked(count) if !write => {
                        count.checked_add(1).unwrap();
                    },
                    _ => {
                        polars_bail!(InvalidOperation: "file already locked");
                    },
                },
                Entry::Vacant(entry) => {
                    if write {
                        entry.insert(State::WriteLocked);
                    } else {
                        entry.insert(State::ReadLocked(NonZeroU32::new(1).unwrap()));
                    }
                },
            }
        }
        Ok(())
    }

    pub fn unlock(file: &mut File) {
        if let Ok(key) = make_key(file) {
            let mut locks = LOCKED_FILES.lock().unwrap();
            match locks.entry(key) {
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    State::WriteLocked => {
                        entry.remove_entry();
                    },
                    State::ReadLocked(count) => {
                        if let Some(new_count) = NonZeroU32::new(count.get() - 1) {
                            *count = new_count;
                        } else {
                            entry.remove_entry();
                        }
                    },
                },
                Entry::Vacant(_) => panic!("attempt to unlock a file that is not locked"),
            }
        }
    }
}

#[cfg(target_family = "windows")]
mod stub {
    // Windows does locking by itself.

    use std::fs::File;

    use polars_error::PolarsResult;

    pub fn lock(file: &mut File, write: bool) -> PolarsResult<()> {
        let _ = file;
        let _ = write;
        Ok(())
    }

    pub fn unlock(file: &mut File) {
        let _ = file;
    }
}

#[cfg(target_family = "unix")]
pub use in_process::{lock, unlock};
#[cfg(target_family = "windows")]
pub use stub::{lock, unlock};
