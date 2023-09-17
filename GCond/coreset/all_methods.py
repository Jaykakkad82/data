import torch
import numpy as np

class Base:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        n = int(data.feat_train.shape[0] * args.reduction_rate) # this is the reduction rate
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def select(self):
        return


    def visualize_embed(self, X,  col, highlight_indices=None, save_path=None, title="t-SNE Visualization"):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn import decomposition
        # PCA
        pca = decomposition.PCA(n_components=2)
        pca.fit(X)
        Xi_train = pca.transform(X)

        # Plot 2D with PCA
        plt.figure(figsize=(8, 6))
        plt.scatter(Xi_train[:, 0], Xi_train[:, 1], c=col, cmap="Set2")
        plt.title("PCA Visualization")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")

        # Highlight select indices (if provided)
        #print("Highlighted indices", highlight_indices )
        #print("X shape", X.shape)
        if highlight_indices is not None:
            X_highlighted = X[highlight_indices]
            Xi_highlighted = Xi_train[highlight_indices]
            plt.scatter(Xi_highlighted[:, 0], Xi_highlighted[:, 1], c='red', label='Highlighted', marker='x')

        plt.legend()
        if save_path is not None:
            save_path1= save_path+"_pca.PNG"
            plt.savefig(save_path1, bbox_inches='tight')

        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=5)
        X_tsne = tsne.fit_transform(X)

        # Plot t-SNE result
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=col, cmap="Set2")
        plt.title("t-SNE Visualization")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Highlight select indices (if provided)
        if highlight_indices is not None:
            X_tsne_highlighted = X_tsne[highlight_indices]
            plt.scatter(X_tsne_highlighted[:, 0], X_tsne_highlighted[:, 1], c='red', label='Highlighted', marker='x')

        plt.legend()

        # Save the plot as a PNG (if save_path is provided)
        if save_path is not None:
            save_path1= save_path+"_tsne.PNG"
            plt.savefig(save_path1, bbox_inches='tight')

        # Optionally
        # plt.show()

        return

class KCenter(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(KCenter, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            # dis = distance(feature, mean)[:,0]
            dis = torch.cdist(feature, mean)[:,0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt-1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)

            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        print("idx_selected : ", idx_selected)
        # visualize
        path = f'visuals_coreset/{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}'
        new = embeds.detach().cpu().numpy()
        self.visualize_embed(new, col=self.data.labels_full, highlight_indices=np.hstack(idx_selected), save_path=path)
        return np.hstack(idx_selected)


class Herding(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Herding, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        # herding # class by class
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        print("idx_selected : ", idx_selected)
        # visualize
        path = f'visuals_coreset/{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}'
        new = embeds.detach().cpu().numpy()
        self.visualize_embed(new, col=self.data.labels_full, highlight_indices=np.hstack(idx_selected), save_path=path )
        return np.hstack(idx_selected)


class Random(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Random, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train

        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        print("idx_selected : ", idx_selected)
        # visualize
        path = f'visuals_coreset/{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}'
        new = embeds.detach().cpu().numpy()
        self.visualize_embed(new  , col= self.data.labels_full, highlight_indices=np.hstack(idx_selected), save_path=path )
        return np.hstack(idx_selected)

class kmeans(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(kmeans, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        # kmeans: class by class
        for class_id, cnt in num_class_dict.items():
            # Select embedding features and labels for the current class
            class_indices = np.where(labels_train == class_id)[0]
            class_embeds = embeds[class_indices].cpu().numpy()
            class_labels = labels_train[class_indices]

            # Determine the number of clusters (cnt) for this class
            num_clusters = min(cnt, len(class_indices))  # Ensure num_clusters doesn't exceed the number of samples

            # Perform K-Means clustering for this class
            from sklearn.cluster import KMeans
            k_means = KMeans(n_clusters=num_clusters)
            k_means.fit(class_embeds)

            # Get cluster assignments and cluster centers
            cluster_assignments = k_means.labels_
            cluster_centers = k_means.cluster_centers_

            # Select representative points (indices) from each cluster
            for cluster_idx in range(num_clusters):
                # Find the indices of samples in this cluster
                cluster_samples_indices = np.where(cluster_assignments == cluster_idx)[0]

                # Calculate the distance of each sample to the cluster center
                cluster_center = cluster_centers[cluster_idx]
                distances = np.linalg.norm(class_embeds[cluster_samples_indices] - cluster_center, axis=1)

                # Select the index of the sample closest to the cluster center
                closest_sample_idx = cluster_samples_indices[np.argmin(distances)]

                # Add the selected index to the list of selected indices
                idx_selected.append(class_indices[closest_sample_idx])

        #return idx_selected
        print("idx_selected : ", idx_selected)
        # return np.array(idx_selected).reshape(-1)
        # visualize
        path = f'visuals_coreset/{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}'
        new = embeds.detach().cpu().numpy()
        self.visualize_embed(new  , col= self.data.labels_full, highlight_indices=np.hstack(idx_selected), save_path=path )
        return np.hstack(idx_selected)

