use crate::gnn::*;
use super::optimizer::Optimizer;

pub struct SGD<'a> {
    pub gnn: &'a mut GNN
}

impl<'a> Optimizer<'a> for SGD<'a> {
    fn new<'b: 'a>(gnn: &'b mut GNN) -> SGD {
        SGD {
            gnn: gnn
        }
    }

    fn train(&mut self, dataset: &Vec<InputData>, batch_size: usize, epochs: usize) {
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
                self.mini_batch(batch_input);
                bar.inc(1);
            }
            bar.finish_and_clear();

            println!("EPOCH: {}, LOSS: {}", e + 1, self.gnn.test_average_loss(&dataset));
        }
    }
}

impl<'a> SGD<'a> {
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
        let (grad_w_avg, grad_a_avg, grad_b_avg) = (grad_w_sum / batch_size, grad_a_sum / batch_size, grad_b_sum / batch_size);

        self.update(grad_w_avg, grad_a_avg, grad_b_avg)
    }

    fn update(&mut self,
           grad_weight: Matrix, grad_a: Vector, grad_bias: f64,
        ) -> (Matrix, Vector, f64) {
        // graph_weight
        let graph_weight_diff = self.gnn.alpha.clone() * grad_weight * (-1.0);

        // output_weight
        let output_weight_diff = self.gnn.alpha.clone() * grad_a * (-1.0);

        // output_bias
        let output_bias_diff = self.gnn.alpha * grad_bias * (-1.0);

        self.gnn.graph_weight = self.gnn.graph_weight.clone() + graph_weight_diff.clone();
        self.gnn.output_weight = self.gnn.output_weight.clone() + output_weight_diff.clone();
        self.gnn.output_bias += output_bias_diff;

        (graph_weight_diff, output_weight_diff, output_bias_diff)
    }
}

