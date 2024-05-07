from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, List, Dict, Tuple

from learning.util import Algorithm, AlgorithmRegistry

Example = Dict[str, Any]
Examples = List[Example]

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class AttrLogicExpression(ABC):
    """
    Abstract base class representing a logic expression.
    """
    ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class Conjunction(AttrLogicExpression):
    """
    A configuration of attribute names and the values the attributes should take for this conjunction to evaluate
    to true.

    `attribute_confs` is a map from attribute names to their values.
    """
    attribute_confs: Dict[str, Any]

    def __post_init__(self):
        assert 'target' not in self.attribute_confs, "Nice try, but 'target' cannot be part of the hypothesis."

    def __call__(self, example: Example):
        """
        Evaluates whether the conjunction applies to an example or not. Returns true if it does, false otherwise.


        Args:
            example: Example to check if the conjunction applies.

        Returns:
            True if the values of *all* attributes mentioned in the conjunction and appearing in example are equal,
            false otherwise.


        """
        return all(self.attribute_confs[k] == example[k] for k in set(self.attribute_confs).intersection(example))

    def __repr__(self):
        return " AND ".join(f"{k} = {v}" for k, v in self.attribute_confs.items())


@dataclass(frozen=True)
class Disjunction(AttrLogicExpression):
    """
    Disjunction of conjunctions.
    """
    conjunctions: List[Conjunction]

    def __call__(self, example: Example):
        """
        Evaluates whether the disjunction applies to a given example.

        Args:
            example: Example to check if the disjunction applies.

        Returns: True if any of its conjunctions returns true, and false if none evaluates to true.

        """
        return any(c(example) for c in self.conjunctions)

    def __repr__(self):
        return " " + "\nOR\n ".join(f"{v}" for v in self.conjunctions)


class Tree(ABC):
    """
    This is an abstract base class representing a leaf or a node in a tree.
    """
    ...


@dataclass
class Leaf(Tree):
    """
    This is a leaf in the tree. It's value is the (binary) classification, either True or False.
    """
    target: bool


@dataclass
class Node(Tree):
    """
    This is a node in the tree. It contains the attribute `attr_name` which the node is splitting on and a dictionary
    `branches` that represents the children of the node and maps from attribute values to their corresponding subtrees.
    """
    attr_name: str
    branches: Dict[Any, Tree] = field(default_factory=dict)


def same_target(examples: Examples) -> bool:
    """
    This function checks whether the examples all have the same target.

    Args:
        examples: Observations to check

    Returns: Whether the examples all have the same target.
    """
    ei = examples[0]
    return all(e['target'] == ei['target'] for e in examples)
    # raise NotImplementedError()


def plurality_value(examples: Examples) -> bool:
    """
    This function returns whether there are more positively or negatively classified examples in the dataset.
    Args:
        examples: Examples to check.

    Returns: True if more examples classified as positive, False otherwise.
    
    """

    true = 0
    false = 0
    for e in examples:
        if(e['target']):
            true += 1
        else:
            false += 1

    # raise NotImplementedError()
    return true >= false


def binary_entropy(examples: Examples) -> float:
    """
    Calculates the binary (shannon) entropy of a dataset regarding its classification.
    Args:
        examples: Dataset to calculate the shannon entropy.

    Returns: The shannon entropy of the classification of the dataset.

    """

    true = 0.0
    false = 0.0

    for e in examples:
        if(e['target']):
            true += 1
        else:
            false += 1
    
    total = true + false

    if (true == 0.0 and false == 0.0):
         return 0.0
    elif true == 0.0:
        return -(( false/(total) ) * math.log(false/(total), 2))
    elif false == 0.0:

        return -(( true/(total) ) * math.log(true/(total), 2))
    else:
        
        return -(( true/(total) ) * math.log(true/(total), 2) + ( false/(total) ) * math.log(false/(total), 2))


def get_paths(tree: Tree) -> List[List[Tuple[str, Any]]]:

    if (isinstance(tree, Leaf)):
        return [ [('Leaf', tree.target)] ]
    
    paths = None
    concat_paths = []

    for branch in (tree.branches).keys():
        paths = get_paths((tree.branches[branch]))
        for path in paths:
            path.insert(0, (tree.attr_name, branch))
            concat_paths.append(path)
            
    return concat_paths

def to_logic_expression(tree: Tree) -> AttrLogicExpression:
    """
    Converts a Decision tree to its equivalent logic expression.
    Args:
        tree: Tree to convert.

    Returns: The corresponding logic expression consisting of attribute values, conjunctions and disjunctions.

    """
    if (tree == None):
        return Disjunction([])

    paths = get_paths(tree)


    dis = []
    for path in paths:
        path = dict(path)
        if((path['Leaf']) == True):
            path.pop('Leaf')
            dis.append(Conjunction(path))

    return Disjunction(dis)
        


@AlgorithmRegistry.register("dtl")
class DecisionTreeLearner(Algorithm):
    """
    This is the decision tree learning algorithm.
    """

    def find_hypothesis(self) -> AttrLogicExpression:
        tree = self.decision_tree_learning(examples=self.dataset.examples, attributes=self.dataset.attributes,
                                           parent_examples=[])
        return to_logic_expression(tree)

    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        """
        This is the main function that learns a decision tree given a list of example and attributes.
        Args:
            examples: The training dataset to induce the tree from.
            attributes: Attributes of the examples.
            parent_examples: Examples from previous step.

        Returns: A decision tree induced from the given dataset.
        """
        if len(examples) == 0:
            return Leaf(plurality_value(parent_examples))
        
        elif same_target(examples):
            return Leaf(examples[0]['target'])
        
        else:
            att = self.get_most_important_attribute(attributes, examples)
            tree = Node(att, {})
            for value in set([example[att] for example in examples]):
                exs = [example for example in examples if example[att] == value]
                subtree = self.decision_tree_learning(exs, [attr for attr in attributes if attr != att], examples)
                tree.branches[value] = subtree
                
            return tree

    def get_most_important_attribute(self, attributes: List[str], examples: Examples) -> str:
        """
        Returns the most important attribute according to the information gain measure.
        Args:
            attributes: The attributes to choose the most important attribute from.
            examples: Dataset from which the most important attribute is to be inferred.

        Returns: The most informative attribute according to the dataset.

        """
        list = [(attribute, self.information_gain(examples, attribute)) for attribute in attributes]
        return max(list, key=lambda x: x[1])[0]
    

    def information_gain(self, examples: Examples, attribute: str) -> float:
        """
        This method calculates the information gain (as presented in the lecture)
        of an attribute according to given observations.

        Args:
            examples: Dataset to infer the information gain from.
            attribute: Attribute to infer the information gain for.

        Returns: The information gain of the given attribute according to the given observations.

        """
        h_examples = binary_entropy(examples)
        h_examples_attr = 0.0

        for value in set([example[attribute] for example in examples]):
            examples_v = [example for example in examples if example[attribute] == value]
            h_examples_attr += (len(examples_v) / len(examples)) * binary_entropy(examples_v)

        ig = h_examples - h_examples_attr
        
        return ig


@AlgorithmRegistry.register("your-algo-name")
class MyDecisionTreeLearner(DecisionTreeLearner):
    ...
