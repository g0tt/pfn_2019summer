use crate::gnn::*;
use super::optimizer::Optimizer;
use std::ops::Mul;

pub struct Adam<'a> {
    pub gnn: &'a mut GNN
}

impl<'a> Optimizer<'a> for Adam<'a> {
    fn new<'b: 'a>(gnn: &'b mut GNN) -> Adam {
        Adam {
            gnn: gnn
        }
    }

    fn train(&mut self, dataset: &Vec<InputData>, batch_size: usize, epochs: usize) {
        let alpha = 0.001;
        let beta_1 = 0.9;
        let beta_2 = 0.999;
        let epsilon = 1e-8;

        let shape = self.gnn.graph_weight.shape();
        let mut m_w_old = Matrix::zeros((shape[0], shape[1]));
        let mut m_a_old = Vector::zeros(self.gnn.output_weight.shape()[0]);
        let mut m_b_old = 0f64;
        let mut v_w_old = Matrix::zeros((shape[0], shape[1]));
        let mut v_a_old = Vector::zeros(self.gnn.output_weight.shape()[0]);
        let mut v_b_old = 0f64;
        let mut beta_1_t = beta_1.clone();
        let mut beta_2_t = beta_2.clone();

        for e in 0..epochs {
            let bar = ProgressBar::new((dataset.len() / batch_size as usize) as u64);
            bar.set_style(ProgressStyle::default_bar().progress_chars("#>-"));
            let mut memory = dataset.clone();
            while memory.len() > 0 {
                let mut batch_input: Vec<InputData> = Vec::new();
                for _i in 0..batch_size {
                    if memory.len() == 0 {
                        break;
                    }
                    let r = rand::random::<usize>() % memory.len(); // get 1 from memory
                    batch_input.push(memory.remove(r));
                }
                let (grad_w_avg, grad_a_avg, grad_b_avg) = self.mini_batch(batch_input);
                let m_w = beta_1 * m_w_old.clone() + (1.0 - beta_1) * grad_w_avg.clone();
                let m_a = beta_1 * m_a_old.clone() + (1.0 - beta_1) * grad_a_avg.clone();
                let m_b = beta_1 * m_b_old + (1.0 - beta_1) * grad_b_avg;
                let v_w = beta_2 * v_w_old.clone() + (1.0 - beta_2) * (grad_w_avg.clone().mul(grad_w_avg.clone()));
                let v_a = beta_2 * v_a_old.clone() + (1.0 - beta_2) * (grad_a_avg.clone().mul(grad_a_avg.clone()));
                let v_b = beta_2 * v_b_old + (1.0 - beta_2) * (grad_b_avg * grad_b_avg);
                let m_hat_w = m_w.clone() / (1.0 - beta_1_t);
                let m_hat_a = m_a.clone() / (1.0 - beta_1_t);
                let m_hat_b = m_b / (1.0 - beta_1_t);
                let v_hat_w = v_w.clone() / (1.0 - beta_2_t);
                let v_hat_a = v_a.clone() / (1.0 - beta_2_t);
                let v_hat_b = v_b / (1.0 - beta_2_t);
                let mut denom_w = v_hat_w.clone();
                let mut denom_a = v_hat_a.clone();
                for x in denom_w.iter_mut() {
                    *x = x.sqrt() + epsilon;
                }
                for x in denom_a.iter_mut() {
                    *x = x.sqrt() + epsilon;
                }
                let denom_b = v_hat_b.sqrt() + epsilon;

                // update
                self.gnn.graph_weight = self.gnn.graph_weight.clone() - alpha * (m_hat_w / denom_w);
                self.gnn.output_weight = self.gnn.output_weight.clone() - alpha * (m_hat_a / denom_a);
                self.gnn.output_bias = self.gnn.output_bias - alpha * (m_hat_b / denom_b);

                m_w_old = m_w.clone();
                m_a_old = m_a.clone();
                m_b_old = m_b.clone();
                v_w_old = v_w.clone();
                v_a_old = v_a.clone();
                v_b_old = v_b.clone();
                beta_1_t *= beta_1;
                beta_2_t *= beta_2;

                bar.inc(1);
            }
            bar.finish_and_clear();

            println!("EPOCH: {}, LOSS: {}", e + 1, self.gnn.test_average_loss(&dataset));
        }
    }
}

impl<'a> Adam<'a> {
    fn mini_batch(&mut self, batch_input: Vec<InputData>) -> (Matrix, Vector, f64) {
        let batch_size = batch_input.len() as f64;
        let shape = self.gnn.graph_weight.shape();
        let mut grad_w_sum = Matrix::zeros((shape[0], shape[1]));
        let mut grad_a_sum = Vector::zeros(self.gnn.output_weight.shape()[0]);
        let mut grad_b_sum = 0f64;
        for data in batch_input {
            self.gnn.graph = data.graph;
            self.gnn.init_vector();
            self.gnn.aggregate(2);
            let (grad_w, grad_a, grad_b) = self.gnn.gradient(data.label);
            grad_w_sum = grad_w_sum.clone() + &grad_w;
            grad_a_sum = grad_a_sum.clone() + &grad_a;
            grad_b_sum = grad_b_sum + &grad_b;
        }
        (grad_w_sum / batch_size, grad_a_sum / batch_size, grad_b_sum / batch_size)
    }
}

