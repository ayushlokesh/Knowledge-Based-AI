include('SET001-0.ax').
fof(a, axiom, ![X]: ( 
                    ![Y]: 
                        ( 
                            member(Y, dirimage(X)) <=> (?[Z]:( member(Z, X) & role(Z,Y) ))  
                        )
                    )
).



fof(b, axiom,  ![X]: (
                      ![Y]: ( 
                                member(Y, valres(X)) <=> (![Z]: (role(Y,Z) => member(Z, X)))  
                            ) 
                      )
).

fof(c, definition, ![X,Y]: (
                                imp_subset(X, Y) <=> (![Z]: (member(Z, X) => member(Z,Y))   ) 
                            )
).

fof(e, conjecture, ![X,Y]:(
                            imp_subset(dirimage(X), Y) <=> imp_subset(X, valres(Y))
                           )
).