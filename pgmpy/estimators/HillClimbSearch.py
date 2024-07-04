from ML import ML_models
import copy
import pandas as pd
#!/usr/bin/env python
from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
    CompositeScore
)

 
class HillClimbSearch(StructureEstimator):
    """
    Class for heuristic hill climb searches for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                old_parents = model.get_parents(Y)
                if old_parents == []:
                    m0 = None
                    # TODO добавить перебор по всем допустимым моделям ML
                    ml_models = ML_models(self.data)
                    # M1 - список допустимых моделей для узла Y
                    M1 = ml_models.get_all_models_by_children_type(Y)                    
                    # if pd.api.types.is_string_dtype(self.data[Y]):
                    #     m1 = 'LogisticRegression'
                    # else:
                    #     m1 = 'LinearRegression'
                else:
                    m0 = model.nodes[Y]['ML_model']
                    M1 = [model.nodes[Y]['ML_model']]
                for m1 in M1:
                    operation = ("+", (X, Y, m1))                
                    if (
                        (operation not in tabu_list)
                        and ((X, Y) not in black_list)
                        and ((X, Y) in white_list)
                    ):

                        new_parents = old_parents + [X]
                        if len(new_parents) <= max_indegree:
                            new = score(Y, new_parents, m1)
                            prev = score(Y, old_parents, m0)
                            score_delta = new - prev
                            score_delta += structure_score("+")
                            yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y, model.nodes[Y]['ML_model']))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = [var for var in old_parents if var != X]
                if new_parents == []:
                    m0 = model.nodes[Y]['ML_model']
                    m1 = None
                else:
                    m0 = model.nodes[Y]['ML_model']
                    m1 = model.nodes[Y]['ML_model']
                operation = ("-", (X, Y, m1))
                score_delta = score(Y, new_parents, m1) - score(Y, old_parents, m0)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                
                old_X_parents = model.get_parents(X)
                old_Y_parents = model.get_parents(Y)
                new_X_parents = old_X_parents + [Y]
                new_Y_parents = [var for var in old_Y_parents if var != X]
                m_x_0 = model.nodes[X]['ML_model']
                m_y_0 = model.nodes[Y]['ML_model']
                if old_X_parents == [] and new_X_parents != []:
                    ml_models = ML_models(self.data)
                    M_x_1 = ml_models.get_all_models_by_children_type(X)  
                else:
                    M_x_1 = [model.nodes[X]['ML_model']]
                if old_Y_parents != [] and new_Y_parents == []:
                    m_y_1 = None
                else:
                    m_y_1 = model.nodes[Y]['ML_model']
                if old_X_parents != [] and new_X_parents != [] and old_Y_parents != [] and new_Y_parents != []:
                    M_x_1 = [model.nodes[X]['ML_model']]
                    m_y_1 = model.nodes[Y]['ML_model']
                for m_x_1 in M_x_1:
                    operation = ("flip", (X, Y, m_x_1, m_y_1))
                    if (
                        ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                        and ((X, Y) not in fixed_edges)
                        and ((Y, X) not in black_list)
                        and ((Y, X) in white_list)
                    ):
                        
                        if len(new_X_parents) <= max_indegree:
                            score_delta = (
                                score(X, new_X_parents, m_x_1)
                                + score(Y, new_Y_parents, m_y_1)
                                - score(X, old_X_parents, m_x_0)
                                - score(Y, old_Y_parents, m_y_0)
                            )
                            score_delta += structure_score("flip")
                            yield (operation, score_delta)

        # Step 4: Get all legal operations for changing ML models
        nodes_with_models = [n for n in model.nodes if model.in_degree(n) > 0] 
        for node in nodes_with_models:
            parents = model.get_parents(node)
            ml_models = ML_models(self.data)
            model_list = ml_models.get_all_models_by_children_type(node)
            try:
                model_list.remove(model.nodes[node]['ML_model'])
            except:
                pass
            for m in model_list:
                operation = ("ML", (node, m))
                new_node = copy.deepcopy(model.nodes[node])
                new_node['ML_model'] = m
                # score_delta = score(new_node, parents) - score(model.nodes[node], parents)
                score_delta = score(new_node['name'], parents, new_node['ML_model']) - score(model.nodes[node]['name'], parents, model.nodes[node]['ML_model'])
                yield (operation, score_delta)




    def estimate(
        self,
        scoring_method="k2score",
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        epsilon=1e-4,
        max_iter=1e6,
        show_progress=True,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None

        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data)
        >>> best_model = est.estimate(scoring_method=BicScore(data))
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        OutEdgeView([('B', 'J'), ('A', 'J')])
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        OutEdgeView([('J', 'A'), ('B', 'J')])
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }
        if (
            (
                isinstance(scoring_method, str)
                and (scoring_method.lower() not in supported_methods)
            )
        ) and (not isinstance(scoring_method, StructureScore)):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        if self.use_cache:
            score_fn = ScoreCache.ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        # Step 1.2: Check the start_dag
        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)
            nx.set_node_attributes(start_dag, None, 'ML_model') # 'LogisticRegression'
            nx.set_node_attributes(start_dag, None, 'name')
            for n in start_dag.nodes: 
                start_dag.nodes[n]['name'] = n

        elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            print(1)
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)

            nodes_with_parents = set(edge[1] for edge in start_dag.edges)
            for node in nodes_with_parents:
                if node in self.data.attrs['str_columns']:
                    start_dag.nodes[node]['ML_model'] = 'CatBoostClassifier' # 'LogisticRegression'
                else:
                    start_dag.nodes[node]['ML_model'] = 'CatBoostRegressor' # 'LinearRegression'

            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag

        if show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        score_list = []
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )
            
            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*(best_operation[1][:2]))
                current_model.nodes[best_operation[1][1]]['ML_model'] = best_operation[1][2]
                # if current_model.nodes[best_operation[1][1]]['ML_model'] == None:
                #     current_model.nodes[best_operation[1][1]]['ML_model'] = 'LogisticRegression'
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1][:2])
                current_model.nodes[best_operation[1][1]]['ML_model'] = best_operation[1][2]
                # if nx.ancestors(current_model, best_operation[1][1]) == set():
                #     current_model.nodes[best_operation[1][1]]['ML_model'] = None                    
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X, Y = best_operation[1][:2]
                current_model.remove_edge(X, Y)
                # if nx.ancestors(current_model, Y) == set():
                #     current_model.nodes[Y]['ML_model'] = None                       
                current_model.add_edge(Y, X)
                # if current_model.nodes[X]['ML_model'] == None:
                #     current_model.nodes[X]['ML_model'] = 'LogisticRegression'          
                m_x = best_operation[1][2]
                m_y = best_operation[1][3]     
                current_model.nodes[X]['ML_model'] = m_x
                current_model.nodes[Y]['ML_model'] = m_y
                tabu_list.append(best_operation)
            elif best_operation[0] == "ML":
                node, model = best_operation[1]
                current_model.nodes[node]['ML_model'] = model
        
            s_m = CompositeScore(data=self.data)
            current_score = s_m.score(current_model)
            score_list.append(current_score)

        # textfile = open('results\\bnlearn' + '\\sangiovese_' + 'convergence.txt', "a")
        # textfile.write(str(score_list))
        # textfile.close()  

        # Step 3: Return if no more improvements or maximum iterations reached.
        return current_model
