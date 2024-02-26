fof(distinct_rooms, axiom, $distinct(r1,r2,r3,r4,r5,r6)).
fof(meaning_of_next, axiom, ![X,Y] : (next(X,Y) <=> ((X = r1 & Y = r2) | (X = r2 & Y = r3) |
                                 (X = r3 & Y = r4) | (X = r4 & Y = r5) | (X = r5 & Y = r6)))).
fof(a_room_can_be_occupied_by_a_single_animal, axiom, ![X] : ((cat(X) => (~dog(X) & ~hamster(X))) &
                                                        (dog(X) => (~cat(X) & ~hamster(X))) &
                                                        (hamster(X) => (~dog(X) & ~cat(X))))).
fof(hamster_is_in_r6, axiom, (hamster(r6)) & (![X] : ((X != r6) <=> (cat(X) | dog(X))))).
fof(room_is_lit, axiom, ![X,Y,Z] : (lit(X) <=>
                                 ( X != r6 & next(Y,X) & next(X,Z)
                                  & ((dog(X) & (dog(Y) & dog(Z)))
                                  | (cat(X) & (cat(Y) | cat(Z))))))).
fof(only_one_room_lit, axiom, ?[X] : (lit(X) & ![Y] : (lit(Y) => Y = X))).