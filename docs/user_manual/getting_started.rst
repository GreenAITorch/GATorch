Getting started with GATorch
============================

To use GATorch simply create a ``GA`` object. You can then track the measurements of a pytorch model by simply using ``attach_model()``. With 
the latter function you can pass a model to your ``GA`` object, which will then measure the energy consumption of the model during each 
forward and backward pass. 

.. code:: python3

    from GA import GA

    # Create the profiler object and attach a model to it
    ga_measure = GA()
    ga_measure.attach_model(model)

Once the model is attached to the ``GA`` object you can simply follow you normal training loop routine. GATorch will take care to create the measurements
in the background. You can then retrieve the measurements at any point. 

.. code:: python3

    # Let's try to do a single forward pass and a backward pass
    x = torch.zeros([1, 1, 28, 28]).to(device)
    y = torch.zeros([1, 10]).to(device)
    pred = model(x) # forward

    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward() # backward
    optimizer.step()

    # Now lets print the mean measurements
    print(ga_measure.get_mean_measurements())

Retrieving the measurements 
---------------------------

To retrieve the different energy consumption measurements you can use either of the following functions:

.. code:: python3
    
    ga_measure.get_mean_measurements()
    ga_measure.get_sum_measurements()
    ga_measure.get_full_measurements()

Respectively they will show the mean energy measurement of a forward or backward pass of the model, the sum of all the measurements for each pass 
and the full list of measurements. The measurements are displayed in Joules. 

You can also convert the readings into a ``pandas.DataFrame``.

.. code:: python3
    
    ga_measure.to_pandas()

+---------------------------+----------------------------+
| NeuralNetwork_forward     | NeuralNetwork_backward     | 
+===========================+============================+
| 0.031860                  | 0.042359                   |
+------------------+------------------+------------------+
| 0.042236                  | 0.034667                   | 
+------------------+------------------+------------------+

To now visualize the results use :mod:`visualize_data()`.

.. code:: python3

    ga_measure.visualize_data(filename='image.png')
    
.. image:: images/total.png
   :width: 600
