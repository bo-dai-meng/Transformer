import argparse
import dataset
import model_train


parser = argparse.ArgumentParser(description="Transformer_2023")

parser.add_argument( '--input_dim', type= int, default= 8, required=True)
parser.add_argument( '--batch_sizes', type= int, default= 1714, required=True)
parser.add_argument( '--epoch', type= int, default= 3000, required=True)
parser.add_argument( '--max_seq_len', type= int, default= 512, required=True)
parser.add_argument( '--num_layers', type= int, default= 8, required=True)
parser.add_argument( '--time_step', type= int, default= 10, required=True)
parser.add_argument( '--middle_dim', type= int, default= 12, required=True)
parser.add_argument( '--num_heads', type= int, default= 2, required=True)
parser.add_argument( '--dropout', type= float, default= 0.1, required=True)
parser.add_argument( '--lr', type= float, default= 0.006, required=True)
parser.add_argument( '--weight_decay', type= float, default= 0.01, required=True)
parser.add_argument('--train_path', type= str, default= 'D:\\Deeplearning\\dl\\Transformer-project02\\Transformer_2023\\Al_ion_battery_data\\1-train.xlsx', required=True)
parser.add_argument('--test_path', type= str, default= 'D:\\Deeplearning\\dl\\Transformer-project02\\Transformer_2023\\Al_ion_battery_data\\1-test.xlsx', required=True)


args = parser.parse_args()

model_train.train(args)
