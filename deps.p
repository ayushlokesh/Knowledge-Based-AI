fof(a, axiom, ![X]:(require(X) => require(depend(X)))).
fof(b, axiom, depend(clang) = llvm).
fof(c, axiom, depend(llvm) = libc).
fof(d, axiom, require(clang)).
fof(e, conjecture, require(libc)).