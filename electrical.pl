
%Ex-1
safe_list([]). %Base case - Empty list is a safe_list.
safe_list([X]) :- component(X). % Base Case - List with a single element is safe_list if that element is a component.
safe_list([Head,Y|Tail]) :- safe_with(Head, Y), safe_list([Head|Tail]),   safe_list([Y|Tail]).    %Step Case of list containing more than one element.

ispart(part(C)) :- component(C). %Checking for a valid part.

isshield(shield([X|_])) :- ispart(X). %Checking for a valid shield.



%Ex-2
safe_design([]). %Base Case - Empty design is a safe_design.
safe_design([part(X)]) :- component(X). % Base Case - List with a single element is safe_design if that element is a valid part.
safe_design([part(X),part(Y)|Tail]) :- safe_with(X,Y), safe_design([part(X)|Tail]), safe_design([part(Y)|Tail]). %Step Case - for 2 parts, follow the similar structure as that of safe_list
safe_design([shield(H)|Tail]) :- isshield(shield(H)), safe_design(H), safe_design(Tail). %Step Case - if a shield and the design are independently safe designs, then shield can be appended to the design.
safe_design([part(X),shield(H)|Tail]) :- safe_design([shield(H)]), safe_design([part(X)|Tail]) . %Step Case - if a shield(H) and the design [part(X)|Tail] are independently safe designs, then [part(X),shield(H)|Tail] is a safe design.


%Ex-3
count_shields([],0). %Base Case - Empty design list contains 0 shields.
count_shields([X| Tail], Count) :- ispart(X), count_shields(Tail, Count). %Step case - if a design contains N shields, then [part(X)|design] also contains N shields.
count_shields([shield(X)| Tail], Count) :- isshield(shield(X)), count_shields(X, M), count_shields(Tail, N), Count is M+N+1. %Step Case - if a design X contains M shields and design T contains N shields, then [shield(X)|T] contains M+N+1 shields.


%Ex-4
split_list([],[],[]). %Base Case - empty list can be splitted into empty list.
split_list([X|Y], L, [X|Z]) :- split_list(Y,L,Z). %Step case - if Y can be splitted into L, Z, then [X|Y] can be splitted into L, [x|Z].
split_list([X|Y], [X|L], Z) :- split_list(Y,L,Z). %Step case - if Y can be splitted into L, Z, then [X|Y] can be splitted into [X|L], Z.

%Ex-4 & Ex-5
design_uses([],[]). %Base case - empty list uses empty list of components.
design_uses([part(H)|T], [X|Y]) :- component(H), split_list([X|Y], [H], Res), design_uses(T,Res). %Step case - If a design A uses a list of components C, and H is a valid component, then a design like [part(H)|A], uses a list of components B such that B can be splitted into A and [H].
design_uses([shield(H)|T], [X|Y]) :- isshield(shield(H)), split_list([X|Y], L, K), design_uses(H, L), design_uses(T,K). %Step case - if a shield S and a design D uses component lists A and B respectively, then a design like [S|D] uses component list C, such that C can be splitted into A, B. 


