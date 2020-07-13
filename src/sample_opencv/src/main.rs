#![feature(conservative_impl_trait, universal_impl_trait)]
extern crate glob;

use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use std::env::*;
use std::path::PathBuf;
use opencv::imgcodecs::*;
use opencv::core::*;
use opencv::objdetect::*;
use opencv::*;
use glob::glob;

struct ImageInfo {
    img: Mat,
    path: PathBuf,
}

impl ImageInfo {
    fn new(img: Mat, path: PathBuf) -> ImageInfo {
        ImageInfo {
            img,
            path,
        }
    }
}

struct FaceDetecter {
    load_handle: Option<JoinHandle<()>>,
    detection_handle: Option<JoinHandle<()>>,
    output_handle: Option<JoinHandle<()>>,
    to_stop_load: Arc<AtomicBool>,
    to_stop_detection: Arc<AtomicBool>,
    to_stop_output: Arc<AtomicBool>,
}

impl FaceDetecter {
    fn new() -> Self {
        FaceDetecter {
            load_handle: None,
            detection_handle: None,
            output_handle: None,

            to_stop_load: Arc::new(AtomicBool::new(false)),
            to_stop_detection: Arc::new(AtomicBool::new(false)),
            to_stop_output: Arc::new(AtomicBool::new(false)),
        }
    }
}

trait FaceDetecterTrait {
    fn run(self);
    fn stop(self);
    fn join(self);
}

impl FaceDetecterTrait for FaceDetecter {
    fn run(mut self) {
        let (tx_load, rx_load): (std::sync::mpsc::Sender<PathBuf>, std::sync::mpsc::Receiver<PathBuf>) = mpsc::channel();
        let (tx_detection, rx_detection): (std::sync::mpsc::Sender<ImageInfo>, std::sync::mpsc::Receiver<ImageInfo>) = mpsc::channel();
        let (tx_output, rx_output): (std::sync::mpsc::Sender<ImageInfo>, std::sync::mpsc::Receiver<ImageInfo>) = mpsc::channel();

        let load = thread::spawn(move || {
            for received in &rx_load {
                /*
                if self.to_stop_load.load(Ordering::Relaxed) {
                    break;
                }
                */

                println!("loadimg image...");
                let path = received.clone();
                let _ = tx_detection.send(ImageInfo::new(load_img(received).unwrap(), path));
                thread::sleep(Duration::from_secs(1));
            }
        });

        let face_detection = thread::spawn(move || {
            for received in &rx_detection {
                /*
                if self.to_stop_detection.load(Ordering::Relaxed) {
                    break;
                }
                */

                println!("FaceDetecter operating...");
                let _ = tx_output.send(ImageInfo::new(face_detection(received.img, ".jpg").unwrap(), received.path));
                thread::sleep(Duration::from_secs(1));
            }
        });

        let output = thread::spawn(move || {
            for received in &rx_output {
                /*
                if self.to_stop_output.load(Ordering::Relaxed) {
                    break;
                }
                */

                println!("Output images...");
                let _ = output_img(received, ".jpg");
                thread::sleep(Duration::from_secs(1));
            }
        });

        self.load_handle = Some(load);
        self.detection_handle = Some(face_detection);
        self.output_handle = Some(output);

        for path in input_filenames(".jpg").unwrap().iter() {
            let _ = tx_load.send(path.to_path_buf());
            thread::sleep(Duration::from_secs(1));
        }

    }

    fn stop(mut self) {
        self.to_stop_load.store(true, Ordering::Relaxed);
        self.to_stop_detection.store(true, Ordering::Relaxed);
        self.to_stop_output.store(true, Ordering::Relaxed);
    }

    fn join(mut self) {
        self.load_handle;
        self.detection_handle;
        self.output_handle;
    }
}


fn load_img(path: PathBuf) -> Result<Mat, ()> {
    Ok(imread(&path.into_os_string().into_string().unwrap(), IMREAD_UNCHANGED).unwrap())
}

fn input_filenames(ext: &str) -> Result<Vec<PathBuf>, std::io::Error> {
    let current_path = current_dir().unwrap();
    let mut input_path: String = String::from(current_path.to_str().unwrap());

    input_path.push_str("/input_images/*");
    input_path.push_str(ext);
    
    let mut v = Vec::new();
    for path in glob(&input_path).unwrap().filter_map(Result::ok) {
        v.push(path);
    }
    Ok(v)
}

fn output_filename(path: PathBuf, ext: &str) -> Result<String, ()> {
    Ok(path.to_str().unwrap().replace("/input_images/", "/output_images/").replace(ext, &format!("{}{}", "_out", ext)))
}

fn output_img(work: ImageInfo, ext: &str) {
    let mut v = Vector::new();
    let _ = v.insert(0, IMWRITE_JPEG_CHROMA_QUALITY);

    let _ = imwrite(&output_filename(work.path, ext).unwrap(), &work.img, &v).unwrap();
}

fn face_detection(img: Mat, ext: &str) -> Result<Mat, opencv::Error> {

    let mut face_cascade: CascadeClassifier = CascadeClassifier::new("/work/opencv/data/haarcascades/haarcascade_frontalface_default.xml").unwrap();

    let mut src_img = Mat::copy(&img)?;
    let mut gray = Mat::default()?;
    let _ = imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0);
    let mut faces = types::VectorOfRect::new();

    let _ = face_cascade.detect_multi_scale(&gray, &mut faces, 1.3, 5, 0, Size_::new(150, 150), Size_::new(150, 150));

    for rect in faces.iter() {
        let _ = imgproc::rectangle(&mut src_img, rect, Scalar_::new(255.0, 0.0, 0.0, 3.0), 1, 8, 0);
    }
    Ok(src_img)
}

fn main() {
    FaceDetecter::new().run();
}

