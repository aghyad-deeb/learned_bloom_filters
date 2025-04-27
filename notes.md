Apr 25 2025
Which one does it make more sense to test? Serving the model locally and doing the whole thing about moving the model and data to GPU and then or have the model loaded on troch serve and then send the data to it and use it?
I think in a production environment, the model is probably already loaded onto the GPU but the data needs to be moved.

Apr 27 
Goals: 
    -  Make sure that the calculations are correct, specifically latency and throughput.
        - [X] checked that the batch inference function works correctly.
    - Remove unnecessary code:
        - [X] Sequential if we can do batch size 1.
        - Unncessary prints

TODO:
    - Fix the problem with incorporating michal's hash function.
    - Run interactive multiple times to store the output.
