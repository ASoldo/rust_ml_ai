use std::convert::TryFrom;
use std::path::Path;

use image::{GenericImageView, imageops::FilterType};
use tch::{
    Device, Kind, Tensor,
    nn::{self, ModuleT, OptimizerConfig},
    vision::{dataset::Dataset, mnist},
};

pub const DIGIT_CLASS_COUNT: i64 = 10;
pub const IMAGE_EDGE_PIXELS: i64 = 28;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
pub struct TrainingConfig<'a> {
    pub data_dir: &'a Path,
    pub model_out: &'a Path,
    pub epochs: i64,
    pub batch_size: i64,
    pub learning_rate: f64,
    pub device: Device,
}

impl<'a> TrainingConfig<'a> {
    pub fn new(data_dir: &'a Path, model_out: &'a Path) -> Self {
        Self {
            data_dir,
            model_out,
            epochs: 5,
            batch_size: 128,
            learning_rate: 1e-3,
            device: Device::cuda_if_available(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub epochs: i64,
    pub final_loss: f64,
    pub test_accuracy: f64,
}

pub fn train_mnist(cfg: &TrainingConfig<'_>) -> Result<TrainingReport> {
    let dataset = mnist::load_dir(cfg.data_dir)?;
    let vs = nn::VarStore::new(cfg.device);
    let net = MnistNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, cfg.learning_rate)?;

    let mut final_loss = 0.0f64;

    for epoch in 1..=cfg.epochs {
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for (images, labels) in dataset
            .train_iter(cfg.batch_size)
            .shuffle()
            .to_device(cfg.device)
        {
            let images = normalize_batch(images).view([-1, IMAGE_EDGE_PIXELS * IMAGE_EDGE_PIXELS]);
            let labels = labels.to_device(cfg.device);

            let logits = net.forward_t(&images, true);
            let loss = logits.cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);

            epoch_loss += loss.double_value(&[]);
            batches += 1;
        }

        if batches > 0 {
            final_loss = epoch_loss / batches as f64;
        }

        println!(
            "Epoch {epoch}/{total} — avg loss: {final_loss:.4}",
            epoch = epoch,
            total = cfg.epochs,
            final_loss = final_loss
        );
    }

    let (correct, total) = evaluate(&net, &dataset, cfg.batch_size, cfg.device);
    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    vs.save(cfg.model_out)?;

    println!(
        "Saved model to {} (test accuracy: {:.3}%)",
        cfg.model_out.display(),
        accuracy * 100.0
    );

    Ok(TrainingReport {
        epochs: cfg.epochs,
        final_loss,
        test_accuracy: accuracy,
    })
}

pub struct DigitClassifier {
    _vs: nn::VarStore,
    net: MnistNet,
    device: Device,
}

impl DigitClassifier {
    pub fn load<P: AsRef<Path>>(model_path: P, device: Device) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let net = MnistNet::new(&vs.root());
        vs.load(model_path)?;
        Ok(Self {
            _vs: vs,
            net,
            device,
        })
    }

    pub fn predict_tensor(&self, input: &Tensor) -> Result<Prediction> {
        let input = input
            .to_device(self.device)
            .to_kind(Kind::Float)
            .view([-1, IMAGE_EDGE_PIXELS * IMAGE_EDGE_PIXELS]);

        let logits = self.net.forward_t(&input, false);
        let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
        let prediction = probs.argmax(-1, false).int64_value(&[0]);
        let probabilities = Vec::<f32>::try_from(probs.squeeze_dim(0))?;

        Ok(Prediction {
            digit: prediction,
            probabilities,
        })
    }

    pub fn predict_image<P: AsRef<Path>>(&self, image_path: P) -> Result<Prediction> {
        let tensor = load_image_tensor(image_path.as_ref(), self.device)?;
        self.predict_tensor(&tensor)
    }
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub digit: i64,
    pub probabilities: Vec<f32>,
}

pub fn predict_image_file<P: AsRef<Path>, Q: AsRef<Path>>(
    model_path: P,
    image_path: Q,
    device: Option<Device>,
) -> Result<Prediction> {
    let device = device.unwrap_or_else(Device::cuda_if_available);
    let classifier = DigitClassifier::load(model_path, device)?;
    classifier.predict_image(image_path)
}

fn evaluate(net: &MnistNet, dataset: &Dataset, batch_size: i64, device: Device) -> (i64, i64) {
    let mut correct = 0i64;
    let mut total = 0i64;

    for (images, labels) in dataset.test_iter(batch_size).to_device(device) {
        let images = normalize_batch(images).view([-1, IMAGE_EDGE_PIXELS * IMAGE_EDGE_PIXELS]);
        let labels = labels.to_device(device);

        let logits = net.forward_t(&images, false);
        let predictions = logits.argmax(-1, false);
        let matches = predictions.eq_tensor(&labels);
        let batch_correct = matches
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]) as i64;

        correct += batch_correct;
        total += labels.size()[0];
    }

    (correct, total)
}

fn normalize_batch(images: Tensor) -> Tensor {
    images.to_kind(Kind::Float) / 255.0
}

fn load_image_tensor(path: &Path, device: Device) -> Result<Tensor> {
    let image = image::open(path)?;
    let image = if image.dimensions() != (IMAGE_EDGE_PIXELS as u32, IMAGE_EDGE_PIXELS as u32) {
        image.resize_exact(
            IMAGE_EDGE_PIXELS as u32,
            IMAGE_EDGE_PIXELS as u32,
            FilterType::CatmullRom,
        )
    } else {
        image
    };

    let gray = image.to_luma8();
    let data: Vec<f32> = gray.pixels().map(|p| p[0] as f32 / 255.0).collect();

    Ok(Tensor::from_slice(&data)
        .to_device(device)
        .view([1, IMAGE_EDGE_PIXELS * IMAGE_EDGE_PIXELS]))
}

#[derive(Debug)]
struct MnistNet {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MnistNet {
    fn new(vs: &nn::Path) -> Self {
        let fc1 = nn::linear(
            vs / "fc1",
            IMAGE_EDGE_PIXELS * IMAGE_EDGE_PIXELS,
            128,
            Default::default(),
        );
        let fc2 = nn::linear(vs / "fc2", 128, DIGIT_CLASS_COUNT, Default::default());
        Self { fc1, fc2 }
    }
}

impl ModuleT for MnistNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.fc1)
            .relu()
            .dropout(0.2, train)
            .apply(&self.fc2)
    }
}
