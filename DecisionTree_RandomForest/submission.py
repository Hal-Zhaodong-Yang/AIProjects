import numpy as np
from collections import Counter
import time

from numpy.lib.index_tricks import c_
from numpy.random.mtrand import f


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    decision_true = DecisionNode(None, None, None, 1)
    decision_false = DecisionNode(None, None, None, 0)
    decision_node5 = DecisionNode(decision_false, decision_true, lambda a: a[3] == 1)
    decision_node4 = DecisionNode(decision_true, decision_false, lambda a: a[3] == 1)
    decision_node3 = DecisionNode(decision_false, decision_node5, lambda a: a[2] == 1)
    decision_node2 = DecisionNode(decision_node3, decision_node4, lambda a: a[1] == 1)
    decision_tree_root = DecisionNode(decision_true, decision_node2, lambda a: a[0] == 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    b_true_labels = np.array(true_labels, dtype = bool)
    b_classifier_output = np.array(classifier_output, dtype = bool)
    c_matrix = [[sum(np.logical_and(b_true_labels, b_classifier_output)), sum(np.logical_and(b_true_labels, np.invert(b_classifier_output)))],\
        [sum(np.logical_and(np.invert(b_true_labels), b_classifier_output)), sum(np.logical_and(np.invert(b_true_labels), np.invert(b_classifier_output)))]]

    return c_matrix


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)


    return c_matrix[0][0]  / (c_matrix[0][0] + c_matrix[1][0])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)
    return c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)

    return (c_matrix[0][0] + c_matrix[1][1]) / len(classifier_output)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    #for binary, we can infer Ig = 1 - p^2 - (1-p)^2
    if len(class_vector) == 0:
        return 0
    
    p = 1.0 * sum(class_vector) / len(class_vector)
    ig = 1 - p ** 2 - (1 - p) ** 2

    return ig


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    #print("pre",previous_classes)
    #print("cur",current_classes)
    ig_pre = gini_impurity(previous_classes)
    ig_everyclass = []
    for item in current_classes:
        ig_everyclass.append(gini_impurity(item) * len(item) / len(previous_classes))
    ig_cur = sum(ig_everyclass)


    return (ig_pre - ig_cur)


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes, self.depth_limit)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        features = np.array(features)
        classes = np.array(classes)
        attr_count = features.shape[1]
        #print(attr_count)
        #print("one node")

        #reach limit
        if gini_impurity(classes) == 0 or depth == 0:
            classes_mean = np.mean(classes)
            if classes_mean < 0.5:
                return DecisionNode(None, None, None, class_label = 0)
            else:
                return DecisionNode(None, None, None, class_label = 1)
        
        ig_gain = []
        threshold = []
        for i in range(attr_count):
            #assume the distribution of class 0 and 1 gaussian
            mean0 = np.mean(features[np.where(classes == 0),i])
            mean1 = np.mean(features[np.where(classes == 1),i])
            #"rough" naive bayes classifier
            split = (mean0 + mean1) / 2
            sub_classes0 = classes[np.where(features[:,i] < split)]
            sub_classes1 = classes[np.where(features[:,i] >= split)]
            threshold.append(split)
            '''
            if len(classes < 10):
                print("features", features[:, i], '\n')
                print("classes", classes, '\n')
                print("subset", (sub_classes0, sub_classes1), '\n')
            '''
            ig_gain.append(gini_gain(classes, [sub_classes0, sub_classes1]))
        
        alpha_best = np.argmax(ig_gain)
        #none of the attribute has any contribution to the result
        if np.max(ig_gain) == 0:
            classes_mean = np.mean(classes)
            if classes_mean < 0.5:
                return DecisionNode(None, None, None, class_label = 0)
            else:
                return DecisionNode(None, None, None, class_label = 1)
        
        #work normally
        sub_classes0 = classes[np.where(features[:,alpha_best] < threshold[alpha_best])]
        sub_features0 = features[np.where(features[:,alpha_best] < threshold[alpha_best])]
        sub_classes1 = classes[np.where(features[:,alpha_best] >= threshold[alpha_best])]
        sub_features1 = features[np.where(features[:,alpha_best] >= threshold[alpha_best])]

        root_node = DecisionNode(None,None,lambda a: a[alpha_best] >= threshold[alpha_best])
        left_node = self.__build_tree__(sub_features1, sub_classes1, depth - 1)
        right_node = self.__build_tree__(sub_features0, sub_classes0, depth - 1)
        root_node.left = left_node
        root_node.right = right_node

        return root_node


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []

        # TODO: finish this.
        for feature in features:
            class_labels.append(self.root.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    features = dataset[0]
    classes = dataset[1]
    remain_indices = list(range(features.shape[0]))
    all_indices = list(range(features.shape[0]))
    n = int(features.shape[0] / k)
    subset_indices = []
    folds = []
    for i in range(k):
        dataset_indices = list(np.random.choice(remain_indices, n, replace = False))
        remain_indices = list(set(remain_indices) - set(dataset_indices))
        subset_indices.append(dataset_indices)
        #print("subset indices",len(set(dataset_indices)))
    
    for i in range(k):
        test_set = (features[subset_indices[i], :], classes[subset_indices[i]])
        remain_indices = list(set(all_indices) - set(subset_indices[i]))
        #print("all count", len(all_indices))
        #print("remain count", len(remain_indices))
        train_set = (features[remain_indices, :], classes[remain_indices])
        folds.append((train_set, test_set))

    #print(len(folds[0][0][1]))
    return folds



class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        features = np.array(features)
        classes = np.array(classes)
        dataset_indices = list(range(features.shape[0]))
        attr_indices = list(range(features.shape[1]))
        for i in range(self.num_trees):
            dataset_sam_idc = np.random.choice(dataset_indices, int(features.shape[0] * self.example_subsample_rate))
            attr_sam_idc = np.sort(np.random.choice(attr_indices, int(features.shape[1] * self.attr_subsample_rate), replace = False))
            sub_features = features[dataset_sam_idc].take(attr_sam_idc, 1)
            sub_classes = classes[dataset_sam_idc]
            tree = DecisionTree(depth_limit = self.depth_limit)
            tree.fit(sub_features, sub_classes)
            self.trees.append((tree, attr_sam_idc))


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        features = np.array(features)
        tree_vote = []
        for tree in self.trees:
            sub_features = features[:, tree[1]]
            tree_vote.append(tree[0].classify(sub_features))
        
        vote_mean = np.mean(tree_vote, axis = 0)
        class_label = np.where(vote_mean > 0.5, 1, 0)

        return class_label


def test_random_forest(k):
    dataset = load_csv('part23_data.csv')
    train_features, train_classes = dataset
    folds = generate_k_folds((train_features, train_classes), k)
    accuracy_list = []
    for fold in folds:
        rf = RandomForest(5, 5, 0.5, 0.5)
        rf.fit(fold[0][0], fold[0][1])
        class_label = rf.classify(fold[1][0])
        accuracy_list.append(accuracy(class_label, fold[1][1]))
    
    print("accuracy ", accuracy_list)
    print("mean accuracy", np.mean(np.array(accuracy_list)))


def test_challenge(k):
    dataset = load_csv('challenge_train.csv', class_index = 0)
    train_features, train_classes = dataset
    folds = generate_k_folds((train_features, train_classes), k)
    #print(folds)
    accuracy_list = []
    for fold in folds:
        rf = ChallengeClassifier()
        rf.fit(fold[0][0], fold[0][1])
        class_label = rf.classify(fold[1][0])
        accuracy_list.append(accuracy(class_label, fold[1][1]))
    
    print("accuracy ", accuracy_list)
    print("mean accuracy", np.mean(np.array(accuracy_list)))

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        self.classifier = RandomForest(15, 10, 0.3, 0.3)

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        self.classifier.fit(features, classes)
        

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        return self.classifier.classify(features)
        


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        

        # TODO: finish this.
        vectorized_data = np.array(data)
        vectorized_data = vectorized_data * vectorized_data + vectorized_data

        return vectorized_data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        sliced_data = np.array(data)[0:100,]
        summed_data = np.sum(sliced_data, axis = 1)
        result = tuple((np.max(summed_data), np.argmax(summed_data)))

        return result

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        #print("non-vectorized", type(unique_dict.items()))

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        occurrence_dist = Counter(np.array(data).flatten())

        #print("vectorized ", occurrence_dist.items())
        key_list = list(occurrence_dist.keys())
        for iter in key_list:
            if iter <= 0:
                del occurrence_dist[iter]

        return occurrence_dist.items()


def return_your_name():
    # return your name
    # TODO: finish this
    return "Zhaodong Yang"
