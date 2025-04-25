Apr 25 2025
Which one does it make more sense to test? Serving the model locally and doing the whole thing about moving the model and data to GPU and then or have the model loaded on troch serve and then send the data to it and use it?
I think in a production environment, the model is probably already loaded onto the GPU but the data needs to be moved.
