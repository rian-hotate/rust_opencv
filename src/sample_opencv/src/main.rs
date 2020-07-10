#![feature(conservative_impl_trait, universal_impl_trait)]
extern crate glob;

use std::sync::mpsc;
use std::thread;

use std::env::*;
use std::path::PathBuf;
use opencv::imgcodecs::*;
use opencv::core::*;
use opencv::objdetect::*;
use opencv::*;
use glob::glob;

struct Worker {
    img: Mat,
    path: PathBuf,
}

impl Worker {
    fn new(img: Mat, path: PathBuf) -> Worker {
        Worker {
            img,
            path,
        }
    }
}

fn main() {
    let (tx_detection, rx_detection): (std::sync::mpsc::Sender<Worker>, std::sync::mpsc::Receiver<Worker>) = mpsc::channel();
    let (tx, rx) =  mpsc::channel();

    thread::spawn(move || {
        println!("Suspending...");
        for received in &rx_detection {
            println!("Output Operating...");
            let _ = output_img(Worker::new(face_detection(received.img, ".jpg").unwrap(), received.path), ".jpg");
            let _ = tx.send(());
        }
    });

    let ext = ".jpg";

    for (path) in input_filenames(ext).unwrap().iter() {
        let img = load_img(path.to_path_buf()).unwrap();
        let _ = tx_detection.send(Worker::new(img, path.to_path_buf()));
        let _ = rx.recv();   
    }
}

fn load_img(path: PathBuf) -> Result<Mat, ()> {
    Ok(imread(&path.into_os_string().into_string().unwrap(), IMREAD_UNCHANGED).unwrap())
}

pub fn input_filenames(ext: &str) -> Result<Vec<PathBuf>, std::io::Error> {
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

pub fn output_filename(path: PathBuf, ext: &str) -> Result<String, ()> {
    Ok(path.to_str().unwrap().replace("/input_images/", "/output_images/").replace(ext, &format!("{}{}", "_out", ext)))
}


fn output_img(work: Worker, ext: &str) {
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
