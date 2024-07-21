import numpy
import scipy.stats

from pyclustering.core.gmeans_wrapper import gmeans as gmeans_wrapper
from pyclustering.core.wrapper import ccore_library

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import distance_metric, type_metric
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans


class CustomGmeans:
    """
    @brief Class implements G-Means clustering algorithm.
    @details The G-means algorithm starts with a small number of centers, and grows the number of centers.
              Each iteration of the G-Means algorithm splits into two those centers whose data appear not to come from a
              Gaussian distribution. G-means repeatedly makes decisions based on a statistical test for the data
              assigned to each center. In this implementation, it uses K-shape for clustering """
    def __init__(self, data, k_init=1, ccore=False, **kwargs):
        """!
        @brief Initializes G-Means algorithm.

        @param[in] data (array_like): Input data that is presented as array of points (objects), each point should be
                    represented by array_like data structure.
        @param[in] k_init (uint): Initial amount of centers (by default started search from 1).
        @param[in] ccore (bool): Defines whether CCORE library (C/C++ part of the library) should be used instead of
                    Python code.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: `tolerance`, `repeat`, `k_max`, `random_state`).

        <b>Keyword Args:</b><br>
            - tolerance (double): tolerance (double): Stop condition for each K-Means iteration: if maximum value of
               change of centers of clusters is less than tolerance than algorithm will stop processing.
            - repeat (unit): How many times K-Means should be run to improve parameters (by default is 3).
               With larger 'repeat' values suggesting higher probability of finding global optimum.
            - k_max (uint): Maximum amount of cluster that might be allocated. The argument is considered as a stop
               condition. When the maximum amount is reached then algorithm stops processing. By default the maximum
               amount of clusters is not restricted (`k_max` is -1).
            - random_state (int): Seed for random state (by default is `None`, current system time is used).
            - significance_level (int): significance level used for statistical testing, by default is 1 (meaning 1%). 2 means (5%) and so on (see Anderson docs)

        """
        self.__data = data
        self.__k_init = k_init

        self.__clusters = []
        self.__centers = []
        self.__centers_kshape = []
        self.__total_wce = 0.0
        self.__ccore = ccore
        self.__empty_clusters = 0
        self.__rep_empty = 0

        self.__tolerance = kwargs.get('tolerance', 0.001)
        self.__repeat = kwargs.get('repeat', 3)
        self.__k_max = kwargs.get('k_max', -1)
        self.__random_state = kwargs.get('random_state', None)
        self.__significance_level = kwargs.get('significance_level', 1)

        if self.__ccore is True:
            self.__ccore = ccore_library.workable()

        self._verify_arguments()



    def _process_by_python(self):
        print('Starting GMEANS')
        """!
        @brief Performs cluster analysis using Python.

        """
        self.__clusters, self.__centers, _, self.__centers_kshape = self._search_optimal_parameters(self.__data, self.__k_init)
        while self._run_condition():
            current_amount_clusters = len(self.__centers)

            self._statistical_optimization()
            proposal_new = current_amount_clusters - len(self.__centers)
            
            # 1st final condition
            if current_amount_clusters == len(self.__centers):  # amount of centers the same - no need to continue.
                break
            #print('final condition is not satisfied. Launching perform clustering... last value=' + str(current_amount_clusters) + ' new value=' + str(len(self.__centers)))
            self._perform_clustering()
            if current_amount_clusters == len(self.__centers):
                #print('final condition satisfied: no updates')
                break
            if (self.__empty_clusters >0):
                #self.__rep_empty = self.__rep_empty + 1
                #if self.__rep_empty == 2:
                # se più della metà dei nuovi cluster proposti sono vuoti (e più di 10)
                #print('final condition satisfied: k-shape is finding empty clusters')
                break

        return self


    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                 collection if 'process()' method was not called.

        """
        nppoints = numpy.array(points)
        if len(self.__clusters) == 0:
            return []

        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE, numpy_usage=True)

        npcenters = numpy.array(self.__centers)
        differences = numpy.zeros((len(nppoints), len(npcenters)))
        for index_point in range(len(nppoints)):
            differences[index_point] = metric(nppoints[index_point], npcenters)

        return numpy.argmin(differences, axis=1)


    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @return (array_like) Allocated clusters.

        @see process()
        @see get_centers()

        """
        return self.__clusters


    def get_centers(self):
        """!
        @brief Returns list of centers of allocated clusters.

        @return (array_like) Allocated centers.

        @see process()
        @see get_clusters()

        """
        return self.__centers


    def get_total_wce(self):
        """!
        @brief Returns sum of metric errors that depends on metric that was used for clustering (by default SSE - Sum of Squared Errors).
        @details Sum of metric errors is calculated using distance between point and its center:
                 \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f]

        @see process()
        @see get_clusters()

        """

        return self.__total_wce


    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION


    def _statistical_optimization(self):
        """!
        @brief Try to split cluster into two to find optimal amount of clusters.

        """
        centers = []
        centers_kshape = [] 
        potential_amount_clusters = len(self.__clusters)
        #print('running statistical optimization. initial potential amount ' + str(potential_amount_clusters))
        for index in range(len(self.__clusters)):
            #print('working on cluster ' + str(index))
            new_centers, new_centers_kshape = self._split_and_search_optimal(self.__clusters[index])
            if (new_centers is None) or ((self.__k_max != -1) and (potential_amount_clusters >= self.__k_max)):
                centers.append(self.__centers[index])
                centers_kshape.append(self.__centers_kshape[index])
                #print(type(self.__centers_kshape[index]))
                #centers_kshape = np.concatenate((centers_kshape, [self.__centers_kshape[index]]))
                #centers_kshape = np.append(centers_kshape, self.__centers_kshape[index], axis=0)
                
            else:
                #print('adding new centers!!!!!!!!')
                centers += new_centers
                #centers_kshape += new_centers_kshape
                # print('centers kshape')
                # print(centers_kshape)
                for j in range(len(new_centers_kshape)):
                    centers_kshape.append(new_centers_kshape[j])
                #print(new_centers)
                #centers_kshape = centers_kshape[0]
                #centers_kshape = np.concatenate((centers_kshape, new_centers_kshape))
                
                # print('centers kshape')
                # print(centers_kshape[0])
                potential_amount_clusters += 1
        
        
        # print('new centers kshape')
        # print(new_centers_kshape)
        # print('centers')
        # print(centers)
        # print('new centers')
        # print(new_centers)
        self.__centers = centers
        self.__centers_kshape = centers_kshape

        #print('centers final')
        #print(self.__centers)
        #print('centers kshape final')
        #print(self.__centers_kshape)
        #print('!!!!!final amount of clusters: ' + str(potential_amount_clusters))
        #print('!!!!!final amount of centers stored: ' + str(len(self.__centers)))
        #print('!!!!!final amount of centers stored (kshape): ' + str(len(self.__centers_kshape)))


        
    def _split_and_search_optimal(self, cluster):
        """!
        @brief Split specified cluster into two by performing K-Means clustering and check correctness by
                Anderson-Darling test.

        @param[in] cluster (array_like) Cluster that should be analysed and optimized by splitting if it is required.

        @return (array_like) Two new centers if two new clusters are considered as more suitable.
                (None) If current cluster is more suitable.
        """
        
        if len(cluster) == 1:
            return None, None

        points = [self.__data[index_point] for index_point in cluster]
        #print('number of points in the considered cluster: ' + str(len(points)))
        new_clusters, new_centers, _ , new_centers_kshape = self._search_optimal_parameters(points, 2)

        if len(new_centers) > 1:
            accept_null_hypothesis = self._is_null_hypothesis(points, new_centers)
            if not accept_null_hypothesis:
                #print('not accept null hypothesis')
                return new_centers, new_centers_kshape  # If null hypothesis is rejected then use two new clusters
        #print('accept null hypothesis')
        return None, None


    def _is_null_hypothesis(self, data, centers):
        """!
        @brief Returns whether H0 hypothesis is accepted using Anderson-Darling test statistic.

        @param[in] data (array_like): N-dimensional data for statistical test.
        @param[in] centers (array_like): Two new allocated centers.

        @return (bool) True is null hypothesis is acceptable.

        """
        v = numpy.subtract(centers[0], centers[1])
        points = self._project_data(data, v)

        estimation, critical, _ = scipy.stats.anderson(points, dist='norm')  # the Anderson-Darling test statistic

        # If the returned statistic is larger than these critical values then for the corresponding significance level,
        # the null hypothesis that the data come from the chosen distribution can be rejected.
        return estimation < critical[-3]  # False - not a gaussian distribution (reject H0)
        # taking critical[-3] we are using 5% as significance level
        


    @staticmethod
    def _project_data(data, vector):
        """!
        @brief Transform input data by project it onto input vector using formula:

        \f[
        x_{i}^{*}=\frac{\left \langle x_{i}, v \right \rangle}{\left \| v \right \|^{2}}.
        \f]

        @param[in] data (array_like): Input data that is represented by points.
        @param[in] vector (array_like): Input vector that is used for projection.

        @return (array_like) Transformed 1-dimensional data.

        """
        square_norm = numpy.sum(numpy.multiply(vector, vector))
        return numpy.divide(numpy.sum(numpy.multiply(data, vector), axis=1), square_norm)


    def _search_optimal_parameters(self, data, amount):
        """!
        @brief Performs cluster analysis for specified data several times to find optimal clustering result in line
                with WCE.

        @param[in] data (array_like): Input data that should be clustered.
        @param[in] amount (unit): Amount of clusters that should be allocated.

        @return (tuple) Optimal clustering result: (clusters, centers, wce).

        """
        #print('running search optimal parameters on ' + str(amount) + ' cluster')
        best_wce, best_clusters, best_centers = float('+inf'), [], []
        best_centers_kshape = []
        for _ in range(self.__repeat):
            # initial_centers = kmeans_plusplus_initializer(data, amount, random_state=self.__random_state).initialize()
            # Initialize cluster centers using k-means++
            #initial_centers = TimeSeriesKMeans(n_clusters=amount, init="k-means++", random_state=self.__random_state).fit(data).cluster_centers

            # solver = kmeans(data, initial_centers, tolerance=self.__tolerance, ccore=False).process()
            #solver = KShape(n_clusters=amount, tol=self.__tolerance)
            #solver.fit(data)
            #print(data)
            n = len(max(data, key=len))
            # Make the lists equal in length
            lst_2 = [x + [0]*(n-len(x)) for x in data]
            a = numpy.nan_to_num(numpy.array(lst_2))
            a = a.reshape(a.shape[0],a.shape[1],1)
            #print('Considering ' + str(a.shape[0]) + ' samples')
            #print(a)
            solver = KShape(n_clusters=amount, random_state=self.__random_state)
            solver.fit(a)
            candidate_wce = solver.inertia_
            empty_cluster=0
            if candidate_wce < best_wce:
                best_wce = candidate_wce
                cluster_labels = solver.labels_
                for i in range(len(solver.cluster_centers_)):
                    indices = numpy.where(cluster_labels == i)[0]
                    if (len(indices)!=0):
                        list_indices = indices.tolist()
                        best_clusters.append(list_indices)
                    else:
                        empty_cluster=empty_cluster+1
                
                if (empty_cluster>0):
                    #print('empty clusters found in search optimal parameters = ' + str(empty_cluster))
                    centers = solver.cluster_centers_[:-empty_cluster]
                    amount = amount - empty_cluster
                else:
                    centers = solver.cluster_centers_
                # Convert to a two-dimensional list
                best_centers = [[item[0] for item in sublist] for sublist in centers]
                # best_centers_kshape = centers.reshape(centers.shape[0]*centers.shape[1],centers.shape[2]) # reshaping for correct input to kshape
                best_centers_kshape = centers
                # best_centers_kshape = np.squeeze(centers, axis=-1)

            if amount == 1: #forse va aggiusto nel caso di cluster empty
                break   # No need to rerun clustering for one initial center.

        return best_clusters, best_centers, best_wce, best_centers_kshape


    def _perform_clustering(self):
        """!
        @brief Performs cluster analysis using K-Means algorithm using current centers are initial.

        @param[in] data (array_like): Input data for cluster analysis.

        """
        # data transformation for Kshape
        n = len(max(self.__data, key=len))
        # Make the lists equal in length
        lst_2 = [x + [0]*(n-len(x)) for x in self.__data]
        a = numpy.nan_to_num(numpy.array(lst_2))
        a = a.reshape(a.shape[0],a.shape[1],1)
        # print(a.shape)
        # print(self.__centers_kshape.shape)
        n_clusters = len(self.__centers_kshape)
        print('perform clustering. I am considering ' + str(n_clusters) + ' clusters')
        # print(self.__centers_kshape)
        #solver = KShape(n_clusters=n_clusters,init=self.__centers_kshape)
        solver = KShape(n_clusters=n_clusters,random_state=self.__random_state)
        solver.fit(a)
        #solver = KShape(n_clusters=n_clusters,init='random').fit(a)
        cluster_labels = solver.labels_ # the output of kshape is the label associated to each sample
        cluster_indices = [] # we need the indices of the data associated with each cluster
        empty_cluster=0
        for i in range(n_clusters):
            indices = numpy.where(cluster_labels == i)[0]
            if (len(indices)!=0):
                list_indices = indices.tolist()
                cluster_indices.append(list_indices)
            else:
                empty_cluster=empty_cluster + 1
        if (empty_cluster>0):
            #print('number of empty clusters: ' +str(empty_cluster))
            #print(len(cluster_indices))
            #self.__clusters = cluster_indices[:-empty_cluster]
            centers_temp = solver.cluster_centers_[:-empty_cluster]
        else:
            #self.__clusters = cluster_indices
            centers_temp = solver.cluster_centers_
        self.__clusters = cluster_indices
        # Convert to a two-dimensional list
        centers = [[item[0] for item in sublist] for sublist in centers_temp]
        self.__centers = centers
        self.__total_wce = solver.inertia_
        # self.__centers_kshape = centers_temp.reshape(centers_temp.shape[0]*centers_temp.shape[1],centers_temp.shape[2])
        self.__centers_kshape = centers_temp
        self.__empty_clusters = empty_cluster
        # self.__centers_kshape = np.squeeze(centers_temp, axis=-1)


    def _run_condition(self):
        """!
        @brief Defines whether the algorithm should continue processing or should stop.

        @return `True` if the algorithm should continue processing, otherwise returns `False`

        """
        if (self.__k_max > 0) and (len(self.__clusters) >= self.__k_max):
            return False

        return True


    def _verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__data))

        if self.__k_init <= 0:
            raise ValueError("Initial amount of centers should be greater than 0 "
                             "(current value: '%d')." % self.__k_init)

        if self.__tolerance <= 0.0:
            raise ValueError("Tolerance should be greater than 0 (current value: '%f')." % self.__tolerance)

        if self.__repeat <= 0:
            raise ValueError("Amount of attempt to find optimal parameters should be greater than 0 "
                             "(current value: '%d')." % self.__repeat)

        if (self.__k_max != -1) and (self.__k_max <= 0):
            raise ValueError("Maximum amount of cluster that might be allocated should be greater than 0 or -1 if "
                             "the algorithm should be restricted in searching optimal number of clusters.")

        if (self.__k_max != -1) and (self.__k_max < self.__k_init):
            raise ValueError("Initial amount of clusters should be less than the maximum amount 'k_max'.")

