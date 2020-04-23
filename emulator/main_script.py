from model import Model
from vpop import Vpop
from emulator import Emulator

descriptors = ["alpha", "beta", "delta", "gamma", "DefaultComp.predator", "DefaultComp.prey"]
vpop = Vpop('vpop_scorer', descriptors=descriptors)
vpop.sample()

model = Model('model score', type='scorer')
simulation_outputs = model.run(vpop)

#emulator = Emulator('emulator', x_parameters=descriptors)

#intput_signal_model_train, output_signal_model_train, _, _ = emulator.format_input(
#    simulation_outputs=simulation_outputs, normalize=False, drop_outliers=False)

#emulator.train(intput_signal_model_train, output_signal_model_train)
