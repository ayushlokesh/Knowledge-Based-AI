fof(distinct_rooms, axiom, $distinct(r1,r2,r3,r4,r5,r6)).

fof(only_single_animal_per_room, axiom, ![X]: (
                                                (cat(X) => (~dog(X) & ~hamster(X)))
                                                &(hamster(X) => (~dog(X) & ~cat(X)))
                                                &(dog(X) => (~cat(X) & ~hamster(X)))
                                               )
).
fof(hamster_in_r6, axiom, hamster(r6)).
fof(next_predicate, axiom, ![X,Y] : (next(X,Y) <=> ((X = r1 & Y = r2) | (X = r2 & Y = r3) |
                                 (X = r3 & Y = r4) | (X = r4 & Y = r5) | (X = r5 & Y = r6)))
                            ).



fof(room_lits, axiom, ![X]: ( lit(X) <=>  
                            ( ?[W,Y]: (
                                        next(W,X) & next(X,Y) & dog(X) & dog(W) & dog(Y)
                                        )
                            |
                              ?[Y]: (
                                        (next(Y,X) | next(X,Y)) & cat(X) & cat(Y)
                                        )   
                            )
                            
                            ) 
).

fof(rooms_for_dogs_cats, axiom, ![X]:( X != r6 => (dog(X) | cat(X)) )).




fof(hamster_never_lit_its_room, axiom, ![X]: (hamster(X) => ~lit(X))).

fof(one_room_lit, axiom, ?[X]: (
                                    lit(X) 
                                    & (![Y]:(lit(Y) <=> Y = X) )
                                )
).

%fof(c, axiom, ~lit(r4)).
%fof(d, axiom, ~lit(r3)).
%fof(e, axiom, ~lit(r2)).

