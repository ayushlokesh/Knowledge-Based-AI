include('SET001-0.ax').
fof(c, axiom, ![X,Y]: (imp_subset(X, Y) <=> (![Z]: (member(Z, X) => member(Z,Y))   )  )   ).

fof(d,conjecture, ![X,Y]: (imp_subset(X,Y) <=> (equal_sets(X,Y)|subset(X,Y))   )    ).