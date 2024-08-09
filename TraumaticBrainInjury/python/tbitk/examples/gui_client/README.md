# Example Usage of the GUI

## Setup

The below instructions assume that tbitk is "installed" into your current virtual environment\.
If this is not the case, follow the instructions in the root ``tbitk/README.md`` file
to set up your environment\.

The below instructions also assume that the user is starting at the root directory of the repository\.

## Getting the GUI Ready
From the root directory of the repository, run:
```
$ pushd TraumaticBrainInjury/python/tbitk/tbitk/ai
$ python gui.py --device_name file_client.py --model_path ../../examples/gui_client/data/duke_study_general_small_8c8772.pt
```

After a matter of seconds, the GUI should pop up on screen. You can also add the flags ``--no_plot_onsd`` to
exclude the plot of the rolling onsd, or ``--separate_source_and_mask`` to create separate displays for the
source and mask frames.

## Getting the Client Ready
In a separate command line window, again starting at the root directory of the repo, run:
```
$ pushd TraumaticBrainInjury/python/tbitk/examples/gui_client
$ python file_client.py --device_name file_client.py --source_path data/e-5.mha
```

You should see the message ``Press Enter to send images.`` after waiting a few seconds.
After pressing enter, the video frames will be sent to the GUI, and the inference will be run.

### Cleanup
After all of the frames have been sent and the GUI is no longer updating, you can hit the "x"
on the GUI to close.

In the client terminal, you should see the message "Press Enter to close."

To return to the root directory, you can run:
```
$ popd
```

In both terminals as well.
