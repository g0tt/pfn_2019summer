extern crate rand;
extern crate ndarray;
extern crate indicatif;

pub mod optimizer;

use ndarray::prelude::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rand::distributions::{Normal, Distribution};
pub use optimizer::Optimizer;

pub type Graph = Array2<i32>;
pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

#[derive(Clone)]
pub struct GNN {
    pub graph: Graph,
    pub graph_weight: Matrix,
    pub feature_vector: Vec<Vector>,
    pub output_weight: Vector,
    pub output_bias: f64,
    pub epsilon: f64,
    pub alpha: f64,
    pub moment: f64,
    pub f: fn(f64) -> f64
}

impl Default for GNN {
    fn default() -> Self {
        GNN {
            graph: Default::default(),
            graph_weight: Default::default(),
            feature_vector: Default::default(),
            output_weight: Default::default(),
            output_bias: 0.0,
            epsilon: 0.001,
            alpha: 0.005,
            moment: 0.9,
            f: relu,
        }
    }
}

impl GNN {
    pub fn aggregate(&mut self, time: u32) -> &GNN {
        for _i in 0..time {
            let mut result: Vec<Vector> = Vec::new();
            for vertex in self.graph.outer_iter() {
                let summation = self.aggregate_1(vertex);
                let new_feature_vector = self.aggregate_2(&summation);
                result.push(new_feature_vector);
            }
            self.feature_vector = result;
        }
        self
    }

    // 集約-1
    pub fn aggregate_1<'b>(
        &self,
        graph: ArrayView1<'b, i32>
    ) -> Vector {
        let mut summation: Vector = Vector::zeros(self.feature_vector[0].len());
        for (k, v) in graph.iter().enumerate() {
            if *v != 0i32 { // 隣接
                summation = summation + self.feature_vector.get(k).unwrap();
            }
        }
        summation
    }

    // 集約-2
    pub fn aggregate_2(&self, summation: &Vector) -> Vector {
        self.graph_weight.dot(summation).mapv(self.f)
    }

    // READOUT
    pub fn readout(&self) -> Vector {
        let mut result = Vector::zeros(self.feature_vector[0].len());
        for v in self.feature_vector.iter() {
            result += v
        }
        result
    }

    // 分類
    pub fn classify(&self, graph: &Graph) -> u8 {
        let mut classifier = self.clone();
        classifier.graph = graph.clone();
        classifier.init_vector();
        classifier.aggregate(2);
        let weighted_sum = classifier.weighted_sum();
        let p = sigmoid(weighted_sum);
        if p > 0.5f64 {
            1
        } else {
            0
        }
    }


    // 特徴ベクトルの初期化
    pub fn init_vector(&mut self) -> &GNN {
        let size = self.graph.len_of(Axis(0));
        let mut x: Vec<Vector> = Vec::new();
        for _i in 0..size {
            // first-hot vector
            x.push(arr1(&(0..self.output_weight.len())
                            .map(|x| if x == 0 { 1f64 } else { 0f64 })
                            .collect::<Vec<f64>>()[..]));
        }
        self.feature_vector = x;
        self
    }

    // 重み和
    pub fn weighted_sum(&self) -> f64 {
        self.output_weight.dot(&self.readout()) + self.output_bias
    }

    // 損失関数
    pub fn loss_function(&self, label: u8) -> f64 {
        let s = self.weighted_sum();
        let y = label as f64;
        if s > 10f64 {
            y * (1.0 + (0.0 - s).exp()).ln() + (1.0 - y) * s
        } else if s < -10f64 {
            y * (0.0 - s) + (1.0 - y) * (1.0 + s.exp()).ln()
        } else {
            y * (1.0 + (0.0 - s).exp()).ln() + (1.0 - y) * (1.0 + s.exp()).ln()
        }
    }

    // 勾配
    pub fn gradient(&self, label: u8) -> (Matrix, Vector, f64) {
        // graph_weight
        let shape = self.graph_weight.shape();
        let mut grad_w = Matrix::zeros((shape[0], shape[1]));
        for (k, v) in self.graph_weight.indexed_iter() {
            let mut after = self.clone();
            {
                let wij = after.graph_weight.get_mut(k).unwrap();
                *wij = v + self.epsilon;
            }
            after.init_vector();
            after.aggregate(2);
            let grad_wij = grad_w.get_mut(k).unwrap();
            *grad_wij = (after.loss_function(label) / self.epsilon) - (self.loss_function(label) / self.epsilon);
        }

        // output_weight
        let mut grad_a = Vector::zeros(self.output_weight.shape()[0]);
        for (k, v) in self.output_weight.indexed_iter() {
            let mut after = self.clone();
            {
                let ai = after.output_weight.get_mut(k).unwrap();
                *ai = v + self.epsilon;
            }
            after.init_vector();
            after.aggregate(2);
            let grad_ai = grad_a.get_mut(k).unwrap();
            *grad_ai = (after.loss_function(label) / self.epsilon) - (self.loss_function(label) / self.epsilon);
        }

        // output_bias
        let mut after = self.clone();
        after.output_bias = self.output_bias + self.epsilon;
        after.init_vector();
        after.aggregate(2);
        let grad_bias = (after.loss_function(label) / self.epsilon) - (self.loss_function(label) / self.epsilon);

        (grad_w, grad_a, grad_bias)
    }

    // Train with single data
    pub fn train(&mut self, label: u8) -> &GNN {
        self.init_vector();
        self.aggregate(2);
        let (weight, a, bias) = self.gradient(label);
        self.update(weight, a, bias)
    }

    fn update(&mut self,
           grad_weight: Matrix, grad_a: Vector, grad_bias: f64,
        ) -> &Self {
        // graph_weight
        let graph_weight_diff = self.alpha.clone() * grad_weight * (-1.0);

        // output_weight
        let output_weight_diff = self.alpha.clone() * grad_a * (-1.0);

        // output_bias
        let output_bias_diff = self.alpha * grad_bias * (-1.0);

        self.graph_weight = self.graph_weight.clone() + graph_weight_diff.clone();
        self.output_weight = self.output_weight.clone() + output_weight_diff.clone();
        self.output_bias += output_bias_diff;

        self
    }

    // datasetに対する精度
    pub fn test_accuracy(&mut self, dataset: &Vec<InputData>) -> f64 {
        let mut correct_count: usize = 0;
        for data in dataset {
            if self.classify(&data.graph) == data.label {
                correct_count += 1;
            }
        }
        correct_count as f64 / dataset.len() as f64
    }

    // datasetに対する平均損失
    pub fn test_average_loss(&mut self, dataset: &Vec<InputData>) -> f64 {
        let mut loss_sum = 0f64;
        for i in 0..dataset.len() {
            self.graph = dataset[i].graph.clone();
            self.init_vector();
            self.aggregate(2);
            let loss = self.loss_function(dataset[i].label);
            loss_sum += loss;
        }
        loss_sum / dataset.len() as f64
    }

}

pub fn relu(x: f64) -> f64 {
    x.max(0.)
}

pub fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (0f64 - x).exp())
}

#[derive(Clone, Debug)]
pub struct InputData {
    pub graph: Graph,
    pub label: u8
}

// 正規分布からサンプリングした行列
pub fn get_square_matrix(size: usize, mean: f64, stddev: f64) -> Matrix {
    let normal = Normal::new(mean, stddev);
    let mut matrix = Matrix::zeros((size, size));
    for mij in matrix.iter_mut() {
        *mij = normal.sample(&mut rand::thread_rng());
    }
    matrix
}

// 正規分布からサンプリングしたベクトル
pub fn get_vector(size: usize, mean: f64, stddev: f64) -> Vector {
    let normal = Normal::new(mean, stddev);
    let mut vector = Vector::zeros(size);
    for vi in vector.iter_mut() {
        *vi = normal.sample(&mut rand::thread_rng());
    }
    vector
}
