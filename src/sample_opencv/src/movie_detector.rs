extern crate glob;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::sync::mpsc::TryRecvError;
use opencv::objdetect::*;
use opencv::imgcodecs::*;
use opencv::core::*;
use opencv::videoio::*;
use opencv::*;

struct MovieInfo {
    movie: videoio::VideoCapture,
    path: PathBuf,
}

impl MovieInfo {
    fn new(movie: videoio::VideoCapture, path: PathBuf) -> MovieInfo {
        MovieInfo {
            movie,
            path,
        }
    }
}

struct FrameInfo {
    img: Mat,
    path: PathBuf,
}

impl FrameInfo {
    fn new(img: Mat, path: PathBuf) -> FrameInfo {
        FrameInfo {
            img,
            path,
        }
    }
}


pub struct MovieFaceDetecter {
    load_handle: Option<JoinHandle<()>>,
    get_frame_handle: Option<JoinHandle<()>>,
    detection_handle: Option<JoinHandle<()>>,
    output_handle: Option<JoinHandle<()>>,

    to_stop_load: Option<std::sync::mpsc::Sender<bool>>,
    to_stop_get_frame: Option<std::sync::mpsc::Sender<bool>>,
    to_stop_detection: Option<std::sync::mpsc::Sender<bool>>,
    to_stop_output: Option<std::sync::mpsc::Sender<bool>>,

    finish_sender: std::sync::mpsc::Sender<bool>,
}

impl MovieFaceDetecter {
    pub fn new(tx: std::sync::mpsc::Sender<bool>) -> Self {
        Self {
            load_handle: None,
            get_frame_handle: None,
            detection_handle: None,
            output_handle: None,

            to_stop_load: None,
            to_stop_get_frame: None,
            to_stop_detection: None,
            to_stop_output: None,

            finish_sender: tx,
        }
    }
}

pub trait MovieFaceDetecterTrait {
    fn run(&mut self, paths: VecDeque<PathBuf>);
    fn stop(&mut self);
    fn join(&mut self);
}

impl MovieFaceDetecterTrait for MovieFaceDetecter {
    fn run(&mut self, mut paths: VecDeque<PathBuf>) {
        let (tx_get_frame, rx_get_frame): (std::sync::mpsc::Sender<MovieInfo>, std::sync::mpsc::Receiver<MovieInfo>) = mpsc::channel();
        let (tx_detection, rx_detection): (std::sync::mpsc::Sender<FrameInfo>, std::sync::mpsc::Receiver<FrameInfo>) = mpsc::channel();
        let (tx_output, rx_output): (std::sync::mpsc::Sender<FrameInfo>, std::sync::mpsc::Receiver<FrameInfo>) = mpsc::channel();

        let tx_get_frame_clone = tx_get_frame.clone();
        let tx_detection_clone = tx_detection.clone();
        let tx_output_clone = tx_output.clone();

        let (tx_to_stop_load, rx_to_stop_load) : (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
        let (tx_to_stop_get_frame, rx_to_stop_get_frame) : (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
        let (tx_to_stop_detection, rx_to_stop_detection) : (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
        let (tx_to_stop_output, rx_to_stop_output) : (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();

        let finish_sender = self.finish_sender.clone();

        let load = thread::spawn(move || loop {
            match rx_to_stop_load.try_recv() {
                Ok(true) | Err(TryRecvError::Disconnected) => {
                    println!("load_handler Terinating...");
                    break;
                }
                _ => {
                    if (paths.len() > 0) {
                        println!("loadimg image...");
                        let path = paths.pop_back().unwrap().to_path_buf();
                        let path_clone = path.clone();
                        let _ = tx_get_frame_clone.send(MovieInfo::new(load_movie(path).unwrap(), path_clone));
                        thread::sleep(Duration::from_secs(1));
                    } else {
                        let _ = finish_sender.send(true);
                        break;
                    }
                }
            }
        });

        let get_frame = thread::spawn(move || loop {
            match rx_to_stop_get_frame.try_recv() {
                Ok(true) | Err(TryRecvError::Disconnected) => {
                    println!("get_frame_handler Terinating...");
                    break;
                }
                _ => {
                    match rx_get_frame.recv() {
                        Ok(mut received) => {
                            println!("get frames...");
                            let mut frame_img = Mat::default().unwrap();
                            let _ = received.movie.read(&mut frame_img);
                            let _ = tx_detection_clone.send(FrameInfo::new(frame_img, received.path));
                            thread::sleep(Duration::from_secs(1));
                        }
                        Err(_) => {
                            println!("Terminating.");
                            break;
                        }
                    }
                }
            }
        });

        let face_detection = thread::spawn(move || loop {
            match rx_to_stop_detection.try_recv() {
                Ok(true) | Err(TryRecvError::Disconnected) => {
                    println!("detection_handler Terinating...");
                    let _ = drop(tx_detection);
                    break;
                }
                _ => {
                    match rx_detection.recv() {
                        Ok(received) => {
                            println!("FrameFaceDetecter operating...");
                            let _ = tx_output_clone.send(FrameInfo::new(face_detection(received.img, ".jpg").unwrap(), received.path));
                            thread::sleep(Duration::from_secs(1));
                        }
                        Err(_) => {
                            println!("Terminating.");
                            break;
                        }
                    }
                }
            }
        });

        let output = thread::spawn(move || loop {
            match rx_to_stop_output.try_recv() {
                Ok(true) | Err(TryRecvError::Disconnected) => {
                    println!("output_handler Terinating...");
                    let _ = drop(tx_output);
                    break;
                }
                _ => {
                    match rx_output.recv() {
                        Ok(received) => {
                            println!("Output images...");
                            let _ = output_img(received, ".jpg");
                            thread::sleep(Duration::from_secs(1));
                        }
                        Err(_) => {
                            println!("Terminating.");
                            break;
                        }
                    }
                }
            }

        });

        self.to_stop_load = Some(tx_to_stop_load);
        self.to_stop_get_frame = Some(tx_to_stop_get_frame);
        self.to_stop_detection = Some(tx_to_stop_detection);
        self.to_stop_output = Some(tx_to_stop_output);

        self.load_handle = Some(load);
        self.get_frame_handle = Some(get_frame);
        self.detection_handle = Some(face_detection);
        self.output_handle = Some(output);
    }

    fn stop(&mut self) {
        if let (Some(to_stop_load)) = (self.to_stop_load.take()) {
            let _ = to_stop_load.send(true);
        }
        if let (Some(to_stop_get_frame)) = (self.to_stop_get_frame.take()) {
            let _ = to_stop_get_frame.send(true);
        }
        if let (Some(to_stop_detection)) = (self.to_stop_detection.take()) {
            let _ = to_stop_detection.send(true);
        }
        if let (Some(to_stop_output)) = (self.to_stop_output.take()) {
            let _ = to_stop_output.send(true);
        }
    }

    fn join(&mut self) {
        if let (Some(load_handle)) = (self.load_handle.take()) {
            let _ = load_handle.join();
        }
        if let (Some(get_frame_handle)) = (self.get_frame_handle.take()) {
            let _ = get_frame_handle.join();
        }
        if let (Some(detection_handle)) = (self.detection_handle.take()) {
            let _ = detection_handle.join();
        }
        if let (Some(output_handle)) = (self.output_handle.take()) {
            let _ = output_handle.join();
        }
    }
}

fn output_filename(path: PathBuf, ext: &str) -> Result<String, ()> {
    //Ok(path.to_str().unwrap().replace("/input_movies/", "/output_movies/").replace(ext, &format!("{}{}", "_out", ext)))
    Ok("output.jpg".to_string())
}

fn output_img(work: FrameInfo, ext: &str) {
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


fn load_movie(path: PathBuf) -> Result<videoio::VideoCapture, ()> {
    Ok(videoio::VideoCapture::from_file(&path.into_os_string().into_string().unwrap(), videoio::CAP_ANY).unwrap())
}