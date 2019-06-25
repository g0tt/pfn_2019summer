use crate::gnn::*;
use super::optimizer::Optimizer;

pub struct MomentumSGD<'a> {
    pub gnn: &'a mut GNN
}

impl<'a> Optimizer<'a> for MomentumSGD<'a> {
    fn new<'b: 'a>(gnn: &'b mut GNN) -> MomentumSGD {
        MomentumSGD {
            gnn: gnn
        }
    }

    fn train(&mut self, dataset: &Vec<InputData>, batch_size: usize, epochs: usize) {
        let shape = self.gnn.graph_weight.shape();
        let mut graph_weight_diff_before = Matrix::zeros((shape[0], shape[1]));
        let mut output_weight_diff_before = Vector::zeros(self.gnn.output_weight.shape()[0]);
        let mut output_bias_diff_before = 0f64;
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
                let result = self.mini_batch(batch_input, graph_weight_diff_before.clone(), output_weight_diff_before.clone(), output_bias_diff_before.clone());
                graph_weight_diff_before = result.0;
                output_weight_diff_before = result.1;
                output_bias_diff_before = result.2;
                bar.inc(1);
            }
            bar.finish_and_clear();

            println!("EPOCH: {}, LOSS: {}", e + 1, self.gnn.test_average_loss(&dataset));
        }
    }
}

impl<'a> MomentumSGD<'a> {
    fn mini_batch(&mut self, batch_input: Vec<InputData>,
            diff_weight_before: Matrix, diff_a_before: Vector, diff_bias_before: f64
        ) -> (Matrix, Vector, f64) {
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

        self.update(grad_w_avg, grad_a_avg, grad_b_avg, diff_weight_before, diff_a_before, diff_bias_before)
    }

    fn update(&mut self,
           grad_weight: Matrix, grad_a: Vector, grad_bias: f64,
           grad_weight_before: Matrix, grad_a_before: Vector, grad_bias_before: f64
        ) -> (Matrix, Vector, f64) {
        // graph_weight
        let mut graph_weight_diff = self.gnn.alpha.clone() * grad_weight * (-1.0);

        // output_weight
        let mut output_weight_diff = self.gnn.alpha.clone() * grad_a * (-1.0);

        // output_bias
        let mut output_bias_diff = self.gnn.alpha * grad_bias * (-1.0);

        graph_weight_diff = graph_weight_diff.clone() + self.gnn.moment.clone() * grad_weight_before;
        output_weight_diff = output_weight_diff.clone() + self.gnn.moment.clone() * grad_a_before;
        output_bias_diff += self.gnn.moment * grad_bias_before;

        self.gnn.graph_weight = self.gnn.graph_weight.clone() + graph_weight_diff.clone();
        self.gnn.output_weight = self.gnn.output_weight.clone() + output_weight_diff.clone();
        self.gnn.output_bias += output_bias_diff;

        (graph_weight_diff, output_weight_diff, output_bias_diff)
    }
}

