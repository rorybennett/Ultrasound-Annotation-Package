### Processes
Special process classes that display something in a spun-out process, since these are more like videos
that can take a fair amount of processing power, which is now in a separate process. There
used to be a bigger need for these, but as the application developed the need vanished. Still
cool to use multiple processes though:
1. `AxisAnglePlot.py`: Display the axis angle of the current Scan object. The current frane position
is also displayed on the plot.
2. `PlayCine.py`: Play a looping recording of the current Scan. Can be slowed down or sped up.