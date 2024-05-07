import itertools
from math import log2
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, astuple
from typing import List, Generator, Union, Tuple

from pyswip import Prolog

from learning.util import Algorithm, Examples, Dataset, Example, AlgorithmRegistry

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class Predicate:
    """
    Object representation of a predicate. Contains `name` which is the name of the predicate and its `arity`.
    """
    name: str
    arity: int

    def __post_init__(self):
        assert self.name[0].islower()


@dataclass(frozen=True)
class Expression:
    """
    Abstract base class representing a valid logical statement.
    """
    ...


@dataclass(frozen=True)
class Literal(Expression):
    """
    Literal: A Predicate with instantiated values for its arguments, which can be either variables or atomic values.

    Converting the literal to string will yield its syntactically valid prolog representation.
    """
    predicate: Predicate = field(hash=True)
    arguments: List[Union['Expression', str]] = field(hash=True)

    def __post_init__(self):
        """
        Make sure that the number of arguments corresponds to the predicate's arity.

        """
        assert (len(self.arguments) == self.predicate.arity,
                f"Number of arguments {len(self.arguments)} not "
                f"equal to the arity of the predicate {self.predicate.arity}")

    def __repr__(self):
        """
        Prolog representation.

        Returns: A syntactically valid prolog representation of the literal.

        """
        return f"{self.predicate.name}({','.join(str(a) for a in self.arguments)})"

    @classmethod
    def from_str(cls, string):
        """
        Generates a python object from a syntactically valid prolog representation.
        Args:
            string: Prolog representation of the literal.

        Returns: `Literal` object equivalent to the prolog representation.

        """
        predicate = get_predicate(string)
        args = get_args(string)
        return Literal(predicate, args)


def get_predicate(text: str) -> Predicate:
    """
    Returns the name and arity of a predicate from a syntactically valid prolog representation.
    Args:
        text: Text to extract the predicate from.

    Returns: Object of `Predicate` class with its corresponding name and arity.

    """
    text = str(text)
    name = text[:text.find("(")].strip()
    arity = len(re.findall("Variable", text))
    if arity == 0:
        arity = len(re.findall(",", text)) + 1
    return Predicate(name, arity)


@dataclass(frozen=True)
class Disjunction(Expression):
    """
    Represents a disjunction of horn clauses which is initially empty.
    """
    expressions: List['HornClause'] = field(default_factory=list)

    def generalize(self, expression: 'HornClause'):
        """
        Adds another horn clause to the disjunction.
        Args:
            expression: Horn clause to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the horn clauses.

        Since there is no real disjunction in prolog, this is just a set of the expressions as separate statements.
        Returns:
            syntactically valid prolog representation of the contained horn clauses.

        """
        return " .\n".join(repr(e) for e in self.expressions) + ' .'


@dataclass(frozen=True)
class Conjunction(Expression):
    """
    Represents a conjunction of literals which is initially empty.
    """
    expressions: List[Expression] = field(default_factory=list)

    def specialize(self, expression: Expression):
        """
        Adds another literal to the conjunction.
        Args:
            expression: literal to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the conjunction of the literals.

        Returns:
            syntactically valid prolog representation of the conjunction (comma-separated).

        """
        return " , ".join(repr(e) for e in self.expressions)


@dataclass(frozen=True)
class HornClause(Expression):
    """
    Represents a horn clause with a literal as `head` and a conjunction as `body`.
    """
    head: Expression
    body: Conjunction = field(default_factory=lambda: Conjunction())

    def get_vars(self):
        """
        Returns all variables appearing in the horn clause.

        Returns: All variables in the horn clause, according to prolog syntax, where variables are capitalised.

        """
        return re.findall(r"(?:[^\w])([A-Z]\w*)", str(self))

    def __repr__(self):
        """
        Converts to a syntactically valid prolog representation.

        Returns:
            Syntactically valid prolog representation of a horn clause in the form of
            ``head :- literal_1 , literal_2 , ... , literal_n``
            for all literals in the body.
        """
        return f"{str(self.head)} :- {' , '.join(str(e) for e in self.body.expressions)}"


def get_args(text: str) -> List[str]:
    """
    Returns the arguments of a text that is assumed to be a single literal in prolog representation.

    Args:
        text: Text to extract the arguments from. Must be valid prolog representation of a single literal.

    Returns:
        All arguments that appear in that literal.

    """
    return [x.strip() for x in re.findall(r"\(.*\)", str(text))[0][1:-1].split(",")]


@AlgorithmRegistry.register('foil')
@contextmanager
def FOIL(dataset: Dataset, recursive=False):
    f = _FOIL(dataset, recursive)
    try:
        yield f
    finally:
        f.abolish()


class _FOIL(Algorithm):
    prolog: Prolog
    var_count: int = -1

    def __init__(self, dataset: Dataset, recursive=False):
        super().__init__(dataset)
        logger.info("Creating prolog...")
        self.prolog = Prolog()

        self.recursive = recursive

        if dataset.kb:
            logger.debug(f"Consulting {self.dataset.kb}")
            self.prolog.consult(self.dataset.kb)

    def abolish(self):
        for p, a in (astuple(a) for a in self.get_predicates()):
            self.prolog.query(f"abolish({p}/{a})")

    def predict(self, example: Example) -> bool:
        return any(self.covers(clause=c, example=example) for c in self.hypothesis.expressions)
        
        
    def get_predicates(self) -> List[Predicate]:
        """
        This method returns all (relevant) predicates from the knowledge base.

        Returns:
            all currently known predicates in the knowledge base that was loaded from the file corresponding to the
            dataset.

        """
        pd = []
        file_sentences = None
        preds=[]
        with open(self.dataset.kb, 'r') as file:
            # Read the entire contents of the file
            file_contents = file.read()

        # Split the file contents on '.' and remove empty strings
            
            file_sentences = [sentence.strip() for sentence in file_contents.split('.') if sentence.strip()]
        sets = set()
        for i in file_sentences:
            sets.add(get_predicate(i.split(":-")[0]))
        
        return list(sets)

    def find_hypothesis(self) -> Disjunction:
        """
        Initiates the FOIL algorithm and returns the final disjunction from the list that is returned by
        `FOIL.foil`.

        Returns: Disjunction of horn clauses that represent the learned target relation.

        """
        positive_examples = self.dataset.positive_examples
        negative_examples = self.dataset.negative_examples

        target = Literal.from_str(self.dataset.target)

        predicates = self.get_predicates()
        assert predicates

        clauses = self.foil(positive_examples, negative_examples, predicates, target)
        return Disjunction(clauses)

    def foil(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
             target: Literal) -> List[HornClause]:
        """
        Learns a list of horn clauses from a set of positive and negative examples which as a disjunction
        represent the hypothesis inferred from the dataset.

        This method is the outer loop of the foil algorithm.

        Args:
            positive_examples: Positive examples for the target relation to be learned.
            negative_examples: Negative examples for the target relation to be learned.
            predicates: Predicates that are allowed in the bodies of the horn clauses.
            target: Signature of the target relation to be learned

        Returns:
            A list of horn clauses that as a disjunction cover all positive and none of the negative examples.

        """
        clauses = []
        count = 0
        while len(positive_examples) != 0:
            count += 1

            clause = self.new_clause(positive_examples, negative_examples, predicates, target)
            positive_examples = [e for e in positive_examples if not self.covers(clause, e)]
            clauses.append(clause)

            # assert count < 1
            
        return clauses

    def covers(self, clause: HornClause, example: Example) -> bool:
        """
        This method checks whether an example is covered by a given horn clause under the current knowledge base.
        Args:
            clause: The clause to check whether it covers the examples.
            example: The examples to check whether it is covered by the clause.

        Returns:
            True if covered, False otherwise

        """
        
        # list(self.prolog.query(str(clause)))
        s = str(clause.body)
        for k in example.keys():
            s = s.replace(str(k), example[k])
        # print(s)
        if len(list(self.prolog.query(s))) == 0:
            return False
        return True

        

        # if any( 
        #         all(q[k] == example[k] for k in example.keys())
        #          for q in list(self.prolog.query(str((clause.body))))):
        #     return True
        # return False
        
        # raise NotImplementedError()

    def new_clause(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
                   target: Literal) -> HornClause:
        """
        This method generates a new horn clause from a dataset of positive and negative examples, a target and a
        list of allowed predicates to be used in the horn clause body.

        This corresponds to the inner loop of the foil algorithm.

        Args:
            positive_examples: Positive examples of the dataset to learn the clause from.
            negative_examples: Negative examples of the dataset to learn the clause from.
            predicates: List of predicates that can be used in the clause body.
            target: Head of the clause to learn.

        Returns:
            A horn clause that covers some part of the positive examples and does not contradict any of the
            negative examples.

        """
        head = target
        body = []
        while len(negative_examples) != 0:
            candidates = self.generate_candidates(HornClause(head, Conjunction(body)), predicates)
            new_literal = self.get_next_literal(candidates, positive_examples, negative_examples)
            body.append(new_literal)
            new_positive_examples = [ex for pos in positive_examples for ex in self.extend_example(pos, new_literal)]
            new_negative_examples = [ex for neg in negative_examples for ex in self.extend_example(neg, new_literal)]
            positive_examples = new_positive_examples
            negative_examples = new_negative_examples
        return HornClause(head, Conjunction(body))

    def get_next_literal(self, candidates: List[Expression], pos_ex: Examples, neg_ex: Examples) -> Expression:
        """
        Returns the next literal with the highest information gain as computed from a given dataset of positive and
        negative examples.
        Args:
            candidates: Candidates to choose the one with the highest information gain from.
            pos_ex: Positive examples of the dataset to infer the information gain from.
            neg_ex: Negative examples of the dataset to infer the information gain from.

        Returns:
            the next literal with the highest information gain as computed
            from a given dataset of positive and negative examples.

        """
        return max(candidates, key=lambda c: self.foil_information_gain(c, pos_ex, neg_ex))

    def foil_information_gain(self, candidate: Expression, pos_ex: Examples, neg_ex: Examples) -> float:
        """
        This method calculates the information gain (as presented in the lecture) of an expression according
           to given positive and negative examples observations.

        Args:
               candidate: Attribute to infer the information gain for.
               pos_ex: Positive examples to infer the information gain from.
               neg_ex: Negative examples to infer the information gain from.

        Returns: The information gain of the given attribute according to the given observations.

        """
        pos_ex_new = []
        for ex in pos_ex:
            pos_ex_new += self.extend_example(ex, candidate)
        neg_ex_new = []
        for ex in neg_ex:
            neg_ex_new += self.extend_example(ex,candidate)
        
        p0 = len(pos_ex)
        n0 = len(neg_ex)
        p1 = len(pos_ex_new)
        n1 = len(neg_ex_new)
        # print(p0, n0, p1, n1)

        t = len([p for p in pos_ex if is_represented_by(p, pos_ex_new)])
        if(p1 == 0):
            return -999999999999
        return t * (log2(p1/(p1 + n1)) -    log2(p0/(p0 + n0)))             
        
    def generate_variable_orderings(self, elements: List[str], extra: List[str], arity: int) -> List[List[str]]:
        

        # Generate all possible combinations with replacement
        all_combinations = []
        max_len =  len(elements)
        if arity < max_len:
            max_len = arity
        for r in range(1, max_len + 1):  # Iterate over different sizes
            # Generate combinations of elements
           
            element_combinations = itertools.combinations_with_replacement(elements, r)
            # Generate combinations of extra variables
            extra_combinations = itertools.combinations_with_replacement(extra,max_len-r)
           
            for extra_combination, element_combination in itertools.product(element_combinations,extra_combinations):
                combination = extra_combination + element_combination
                all_combinations.append(combination)
            
        # Print the generated combinations
        all_candidates = set()
        for combination in all_combinations:
            candidate_permutations = itertools.permutations(combination)
            for perm in candidate_permutations:
                all_candidates.add(perm)

        return [list(candidate) for candidate in all_candidates]
    
    def generate_candidates(self, clause: HornClause, predicates: List[Predicate]) -> Generator[Expression, None, None]:
        """
        This method generates all reasonable (as discussed in the lecture) specialisations of a horn clause
        given a list of allowed predicates.

        Args:
            clause: The clause to calculate possible specialisations for.
            predicates: Allowed predicate vocabulary to specialise the clause.

        Returns:
            All expressions that could be a reasonable specialisation of `clause`.

        """
        max_arity = max(p.arity for p in predicates)
        extra_vars = [self.unique_var() for i in range(max_arity-1)]
        rule_vars = clause.get_vars()
        candidates = []
        for pred in predicates:
            variables = []
            if pred.arity > 0:
                
                variables = self.generate_variable_orderings(rule_vars, extra_vars[:pred.arity-1], pred.arity)
            for var in variables:
                
                candidates.append( Literal(pred, var) )
            
        return candidates

    def extend_example(self, example: Example, new_expr: Expression) -> Generator[Example, None, None]:
        """
        This method extends an example with all possible substitutions subject to a given expression and the current
        knowledge base.
        Args:
            example: Example to extend.
            new_expr: Expression to extend the example with.

        Returns:
            A generator that yields all possible substitutions for a given example an an expression.

        """
        
        # print("******************************************************************")
        s = str(new_expr)
        # print(s)
        for k in example.keys():
            s = s.replace(str(k), example[k])
        # print(s)
        return [dict(itertools.chain(example.items(), q.items())) for q in list(self.prolog.query(str(s)))]

    def unique_var(self) -> str:
        """
        Returns the next uniquely numbered variable to be used.

        Returns:
            the next uniquely named variable in the following format: `V_i` where `i` is a number.

        """
        self.var_count += 1
        return f"V_{self.var_count}"


def is_represented_by(example: Example, examples: Examples) -> bool:
    """
    Checks whether a given example is represented by a list of examples.
    Args:
        example: Example to check whether it's represented.
        examples: Examples to check whether they represent the example.

    Returns:
        True, if for some `e` in `examples` for all variables (keys except target) in `example`,
        the values are equal (potential additional variables in `e` do not need to be considered). False otherwise.

    """
    return any(example.items() <= e.items() for e in examples)
    
