# COMP24412 2023 Labs

This is the base repo for COMP24412 labs in the academic session 2023-24.

This branch holds the materials for the `lab1` assignment.
You can always return to this branch with the command
```
git checkout lab1
```

There is a refresh script to fetch the lab materials when they become available.
You **must** run this script before you start working on the assignment.
This can be done with the command
```
./refresh.sh
```

To submit your work you need to follow the coursework instructions in the
[CS Handbook](https://wiki.cs.manchester.ac.uk/index.php/UGHandbook22:Coursework#Developing_and_submitting_with_Gitlab).
Once you commit your solution files as detailed in the lab manual,
you **must** tag your submission as `lab1_sol` and push this tag to your repo.
This usually involves the following sequence of commands
```
git add -A .
git commit
git tag lab1_sol
git push origin
git push --tags origin
```

The [CS Handbook](https://wiki.cs.manchester.ac.uk/index.php/UGHandbook22:Coursework#Correcting_an_Incorrect_Tag)
also has instructions in case you need to amend your submission and then fix your submission tag.

It is **your responsibility** to ensure that the submission procedure is completed correctly by the appropriate deadline.
Please ask for support in the lab sessions if you're unsure about lab submission and instructions.

[modeline]: # ( vim:set spell spl=en: )
