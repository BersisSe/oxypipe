//! This module contains the asynchronous pipeline to run a model though.

use anyhow::{Result, anyhow};
use tokio::sync::mpsc;
use tokio::task;

use crate::sources::StreamSource;
use crate::types::Frame;

/// Main Pipeline Struct of oxypipe. A Pipeline allows you to add `Stages` to sources. Sources are ran though the stages asyncrnously.
pub struct Pipeline<T> {
    receiver: mpsc::Receiver<Result<T>>,
}

impl<T: Send + 'static> Pipeline<T> {
    pub fn from_source<S: StreamSource>(source: S) -> Pipeline<Frame> {
        Pipeline::<Frame>::from_receiver(source.stream())
    }

    /// Creates a single-item pipeline from an already loaded frame.
    pub fn from_frame(frame: T) -> Self {
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            if tx.send(Ok(frame)).await.is_err() {
                tracing::warn!("Pipeline receiver dropped before the frame was consumed");
            }
        });

        Pipeline { receiver: rx }
    }

    /// Creates a pipeline from a raw receiver of frames.
    pub fn from_receiver(receiver: mpsc::Receiver<Frame>) -> Pipeline<Frame> {
        let (tx, rx) = mpsc::channel(2);

        tokio::spawn(async move {
            let mut receiver = receiver;
            while let Some(frame) = receiver.recv().await {
                if tx.send(Ok(frame)).await.is_err() {
                    break;
                }
            }
        });

        Pipeline { receiver: rx }
    }

    /// Adds a stage to the pipeline.
    /// The stage runs on a blocking thread, strictly sequentially, with backpressure.
    /// Errors short-circuit: the closure is not invoked for failed items.
    pub fn map<U, F>(mut self, f: F) -> Pipeline<U>
    where
        U: Send + 'static,
        F: FnMut(T) -> Result<U> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(2);
        let mut f_slot = Some(f);

        tokio::spawn(async move {
            while let Some(item) = self.receiver.recv().await {
                let mut f = f_slot.take().expect("stage closure slot invariant");

                let outcome = match item {
                    Ok(item) => {
                        let joined = task::spawn_blocking(move || ((f)(item), f)).await;
                        match joined {
                            Ok((result, f_back)) => {
                                f_slot = Some(f_back);
                                result
                            }
                            Err(join_err) => Err(join_error(join_err)),
                        }
                    }
                    Err(e) => Err(e),
                };

                if tx.send(outcome).await.is_err() {
                    break;
                }
            }
        });

        Pipeline { receiver: rx }
    }

    /// Awaits the next item from the pipeline.
    /// `None` means the stream ended, `Some(Err)` means a stage failed for that item.
    pub async fn next(&mut self) -> Option<Result<T>> {
        self.receiver.recv().await
    }
}

fn join_error(join_err: task::JoinError) -> anyhow::Error {
    if join_err.is_panic() {
        anyhow!("Pipeline stage panicked")
    } else {
        anyhow!("Pipeline stage was cancelled")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(id: u64) -> Frame {
        Frame {
            id,
            width: 2,
            height: 2,
            format: crate::types::PixelFormat::Rgb8,
            data: bytes::Bytes::from(vec![0u8; 12]),
            timestamp_ns: 0,
        }
    }

    #[tokio::test]
    async fn from_frame_yields_value_then_none() {
        let mut pipe = Pipeline::from_frame(make_frame(7)).map(|f| Ok(f.id * 10));

        assert_eq!(pipe.next().await.map(|r| r.unwrap()), Some(70));
        assert!(pipe.next().await.is_none());
    }

    #[tokio::test]
    async fn stage_error_propagates_and_skips_downstream() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();

        let mut pipe = Pipeline::from_frame(make_frame(1))
            .map(|_| Err(anyhow!("boom")))
            .map(move |v: u64| {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(v + 1)
            });

        let item = pipe.next().await.expect("stream should yield the error");
        assert!(item.is_err());
        assert!(pipe.next().await.is_none());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn sequential_ordering_is_preserved() {
        let mut pipe = Pipeline::<Frame>::from_receiver({
            let (tx, rx) = mpsc::channel(4);
            for id in 0..4u64 {
                tx.send(make_frame(id)).await.unwrap();
            }
            rx
        })
        .map(|f| Ok(f.id));

        assert_eq!(pipe.next().await.map(|r| r.unwrap()), Some(0));
        assert_eq!(pipe.next().await.map(|r| r.unwrap()), Some(1));
        assert_eq!(pipe.next().await.map(|r| r.unwrap()), Some(2));
        assert_eq!(pipe.next().await.map(|r| r.unwrap()), Some(3));
        assert!(pipe.next().await.is_none());
    }

    #[tokio::test]
    async fn stage_panic_surfaces_as_error() {
        let mut pipe =
            Pipeline::from_frame(make_frame(1)).map(|_| -> Result<u64> { panic!("stage blew up") });

        let item = pipe.next().await.expect("stream should yield join error");
        let err = item.expect_err("panic should become Err");
        assert!(err.to_string().contains("panicked"));
        assert!(pipe.next().await.is_none());
    }
}
