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

fof(e, conjecture, ![X,Y]:(
                                (equal_sets(dirimage(X), Y) | subset(dirimage(X), Y)) <=> (equal_sets(X, valres(Y)) | subset(X, valres(Y)))
                           )
).