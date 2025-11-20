import torch
import torch.nn as nn
from torch.autograd import Variable
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def patch_shuffle(feats, shuffle_ratio=0.5):
    num_patches = feats.shape[0]
    num_to_shuffle = int(num_patches * shuffle_ratio)

    if num_to_shuffle < 2:
        return feats  
    shuffle_indices = np.random.choice(num_patches, num_to_shuffle, replace=False)
    shuffled_feats = feats.copy()
    shuffled_feats[shuffle_indices] = shuffled_feats[np.random.permutation(shuffle_indices)]
    
    return shuffled_feats

def get_bag_feats_train(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'dsmil-wsi/datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    elif args.dataset == 'Camelyon':
        feats_csv_path = 'dsmil-wsi/datasets/Camelyon16/' + csv_file_df.iloc[0].split('/')[2] + '/' + csv_file_df.iloc[0].split('/')[3]
    elif args.dataset == 'CamelyonResNet':
        feats_csv_path = 'dsmil-wsi/datasets/Camelyon16_ResNet/' + csv_file_df.iloc[0].split('/')[2] + '/' + csv_file_df.iloc[0].split('/')[3]
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)  
    feats = feats.to_numpy()  

    feats = patch_shuffle(feats, shuffle_ratio=0.5) 

    if args.augment_randfloat_constant > 0.0:  # Augments the feature matrix feats by adding noise
        feats = feats + np.random.randn(feats.shape[0], feats.shape[1]) * args.augment_randfloat_constant * np.random.randn()
    if args.augment_randfloat_variable > 0.0:
        feats = feats + np.ones((np.shape(feats)[0], np.shape(feats)[1])) * (np.std(feats, 0) * np.random.randn(np.shape(feats)[0], np.shape(feats)[1])) * args.augment_randfloat_variable

    if args.intra_multilinear_combine > 0.0:  # Multilinear combination of two vectors
        if np.random.rand() < args.intra_multilinear_combine:  # Not each vector is combined, 0 none, 1 all
            rand_vec = np.random.rand(feats.shape[1])
            feats = rand_vec * feats + (1 - rand_vec) * shuffle(feats)

    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df.iloc[1]  # Second column of data.csv file contains labels
    else:
        if int(csv_file_df.iloc[1]) <= (len(label) - 1):  # One-hot encoding only when num_classes is > 1
            label[int(csv_file_df.iloc[1])] = 1

    if args.limit_patches_per_wsi > 0:  # Limits the number of patches in a WSI
        feats = feats[0:args.limit_patches_per_wsi, :]

    return label, feats


def get_bag_feats_test(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'dsmil-wsi/datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    elif args.dataset == 'Camelyon':
        feats_csv_path = 'dsmil-wsi/datasets/Camelyon16/' + csv_file_df.iloc[0].split('/')[2] + '/' + csv_file_df.iloc[0].split('/')[3]
    elif args.dataset == 'CamelyonResNet':
        feats_csv_path = 'dsmil-wsi/datasets/Camelyon16_ResNet/' + csv_file_df.iloc[0].split('/')[2] + '/' + csv_file_df.iloc[0].split('/')[3]
    else: # Same as Training
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1

    if args.limit_patches_per_wsi > 0:
        feats = feats[0:args.limit_patches_per_wsi,:]

    return label, feats


def train(train_df, milnet, criterion, optimizer, args):
    milnet.train() 
    csvs = shuffle(train_df).reset_index(drop=True) 
    total_loss = 0
    if device.type == 'cpu':
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor

    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats_train(train_df.iloc[i], args) 

        if args.inter_combine_all > 0:
            if np.random.rand() < args.inter_combine_all:
                label2, feats2 = get_bag_feats_train(train_df.iloc[np.random.randint(0, len(train_df)-1)], args)
                weight = np.random.rand()
                label = (weight * label + (1-weight) * label2) / 2
                feats = (weight * feats + (1-weight) * feats2) / 2

        if args.inter_combine_sameclass > 0:
            if np.random.rand() < args.inter_combine_sameclass:
                label2, feats2 = get_bag_feats_train(train_df.iloc[np.random.randint(0, len(train_df)-1)], args)
                while label2[0] != label[0]:
                    label2, feats2 = get_bag_feats_train(train_df.iloc[np.random.randint(0, len(train_df)-1)], args)

                weight = np.random.rand()
                label = (weight * label + (1-weight) * label2) / 2
                feats = (weight * feats + (1-weight) * feats2) / 2

        feats = dropout_patches(feats, args.dropout_patch) 
        bag_label = Variable(Tensor(np.array(label)))
        bag_feats = Variable(Tensor(np.array(feats)))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats) 
        max_prediction, _ = torch.max(ins_prediction, 0) 
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1)) 
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = (1-args.instance_weight)*bag_loss + args.instance_weight*max_loss 
        loss.backward() 
        optimizer.step()
        total_loss = total_loss + loss.item() 
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p): 
    if p < 0:
        p = np.random.rand()*(1+p) - p 

    idx = np.random.choice(np.arange(feats.shape[0]), int(np.ceil(feats.shape[0]*(1-p))), replace=False) 
    sampled_feats = np.take(feats, idx, axis=0) 
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(np.floor(feats.shape[0]*p)), replace=True) 
    pad_feats = np.take(sampled_feats, pad_idx, axis=0) 
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0) 
    return sampled_feats

def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval() 
    csvs = shuffle(test_df).reset_index(drop=True) 
    total_loss = 0
    test_labels = []
    test_predictions = []

    if device.type == 'cpu':
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor

    classifications = []
    metrics = {}  
    with torch.no_grad(): 
        for i in range(len(test_df)):
            label, feats = get_bag_feats_test(test_df.iloc[i], args) 
            bag_label = Variable(Tensor(np.array(label))) 
            bag_feats = Variable(Tensor(np.array(feats)))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, attention, _ = milnet(bag_feats) 
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = (1 - args.instance_weight) * bag_loss + args.instance_weight * max_loss
            total_loss = total_loss + loss.item() 
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item())) 
            test_labels.extend([label]) 
            test_predictions.extend([(args.instance_weight*torch.sigmoid(max_prediction)+(1+args.instance_weight)*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            classifications.append(np.array((torch.argmax(bag_prediction.view(1, -1)) == torch.argmax(bag_label.view(1, -1))).cpu())) # Classification accuracy

    test_labels = np.array(test_labels) 
    test_predictions = np.array(test_predictions)

    # Calculating classification accuracy
    accuracy = np.mean(np.array(classifications))
    print('\r accuracy: ' + str(np.mean(np.array(classifications))))

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    
    # Threshold predictions to obtain binary values
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='weighted')
    
    # Bag score = accumulates the number of test samples where the predicted label matches the true label
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score += np.array_equal(test_labels[i], test_predictions[i])
    avg_score = bag_score / len(test_df) 

    metrics.update({
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(np.mean(auc_value))
    })
    
    return total_loss / len(test_df), avg_score, auc_value, bag_score, thresholds_optimal, metrics

def multi_label_roc(labels, predictions, num_classes, pos_label=1): 
    fprs = [] 
    tprs = [] 
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c] 
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1) 
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal
 

def optimal_thresh(fpr, tpr, thresholds, p=0): 
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0) 
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--limit_training_data', default=0, type=int, help='amount of data used for training, rest for testing')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--instance_weight', default=0.5, type=float, help='bag weight = 1-instance weight')
    parser.add_argument('--result_string', default='test', type=str, help='results are written to output/ <string> .json')
    parser.add_argument('--num_randsplits', default='2', type=int, help='number of random splits performed')
    parser.add_argument('--augment_randfloat_constant', default=0.0, type=float, help='augmentation with adding random values to features vectors')
    parser.add_argument('--augment_randfloat_variable', default=0.0, type=float, help='augmentation with adding random values to features vectors; rand number is multiplied with standard deviation of the respective dimension')
    parser.add_argument('--intra_multilinear_combine', default=0.0, type=float, help='augmentation with linear combinations of vectors, alpha drawn from uniform distribution')
    parser.add_argument('--intra_linear_combine', default=0.0, type=float, help='augmentation with linear combinations of vectors, alpha drawn from a certain dist')
    parser.add_argument('--inter_combine_all', default=0.0, type=float, help='inter slide linear combination, feature and label vectors are combined')
    parser.add_argument('--inter_combine_sameclass', default=0.0, type=float, help='inter slide linear combination, only WSIs from the same class are combined')
    # Experimental:
    parser.add_argument('--aug_intra', default=np.nan, type=float, help='new augmentation, linear intra slide (redundant with intra_linear_dist, but for all vectors)') # experimental!
    parser.add_argument('--aug_intra_dist', default=np.nan, type=float, help='new augmentation, linear intra slide with distribution (value = sigma)') # experimental!
    parser.add_argument('--aug_intrax', default=np.nan, type=float, help='new augmentation, multilinear intra slide') # experimental!
    parser.add_argument('--aug_intrax_dist', default=np.nan, type=float, help='new augmentation, multilinear intra slide with distribution (value = sigma)') # experimental!
    parser.add_argument('--limit_patches_per_wsi', default=-1, type=int, help='limit the number of patches (features) per wsi')
    args = parser.parse_args()
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    accuracies = []
    metrics_list = []  
    for split_id in range(0, args.num_randsplits):
        ###############Hier Anpassen für Instanz und Embeddet Base (Adjust here for instance- and embedding-based) ########################
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(device)
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).to(device) # Bag-level classifier, dropout rate set to prevent overfitting, nonlinear tells us if activation funcction is linear
        milnet = mil.MILNet(i_classifier, b_classifier).to(device) 

        ###################################################################################

        criterion = nn.BCEWithLogitsLoss() # Loss function for binary classification

        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        if args.dataset == 'TCGA-lung-default':
            bags_csv = 'dsmil-wsi/datasets/tcga-dataset/TCGA.csv'
        elif args.dataset == 'Camelyon':
            bags_csv = 'dsmil-wsi/datasets/Camelyon16/Camelyon16.csv'
        elif args.dataset == 'CamelyonResNet':
            bags_csv = 'dsmil-wsi/datasets/Camelyon16_ResNet/Camelyon16_ResNet.csv'
        else:
            bags_csv = os.path.join('datasets', args.dataset + '/data.csv')

        bags_path = pd.read_csv(bags_csv)

        bags_split_shuffled = shuffle(bags_path, random_state=split_id) 
        if args.limit_training_data == 0:
            train_path = bags_split_shuffled.iloc[0:int(len(bags_split_shuffled)*(1-args.split)), :]
            test_path = bags_split_shuffled.iloc[int(len(bags_split_shuffled)*(1-args.split)):, :]
        else:
            train_path = bags_split_shuffled.iloc[0:args.limit_training_data, :]
            test_path = bags_split_shuffled.iloc[args.limit_training_data:, :]

        print(train_path)
        print(test_path)

        save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y")) 
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth'))) 
        for epoch in range(0, args.num_epochs):
            train_path = shuffle(train_path).reset_index(drop=True)
            test_path = shuffle(test_path).reset_index(drop=True)
            train_loss_bag = train(train_path, milnet, criterion, optimizer, args) 
            if epoch == args.num_epochs-1:
                test_loss_bag, avg_score, aucs, bag_score, thresholds_optimal, metrics = test(test_path, milnet, criterion, optimizer, args) # Test function call
                metrics_list.append(metrics)  
                current_score = (sum(aucs) + avg_score)/2
                if args.dataset=='TCGA-lung':
                    print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' %
                          (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
                else: 
                    print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, BAG_Score %.4f, current_score %.4f, AUC: ' %
                          (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, bag_score, current_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))

            scheduler.step()
            save_name = os.path.join(save_path, str(run+1)+'.pth')

        accuracies.append(metrics['accuracy'])
    print('end')
    data = {
        'metrics': metrics_list  
    }

    with open('output/' + args.result_string + '.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    main()
