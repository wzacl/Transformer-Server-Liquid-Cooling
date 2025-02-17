class PID(object):
    """A simple PID controller."""

    def __init__(
        self,
        Kp=1.0,
        Ki=0.0,
        setpoint=0,
        output_limits=(None, None)):
        
        self.Kp, self.Ki = -Kp, -Ki
        self.setpoint = setpoint
        self._min_output, self._max_output = None, None
        self._proportional = 0
        self._integral = 0
        


    def __call__(self, input_):
        """
        Update the PID controller.

        Call the PID controller with *input_* and calculate and return a control output if
        sample_time seconds has passed since the last update. If no new output is calculated,
        return the previous output instead (or None if no value has been calculated yet).

        :param dt: If set, uses this value for timestep instead of real time. This can be used in
            simulations when simulation time is different from real time.
        """
        err = self.setpoint - input_
        
        self._proportional = self.Kp * err
        
        self._integral += self.Ki * err
        if self._proportional + self._integral < self._min_output:
            self._integral = self._min_output - self._proportional
        elif self._proportional + self._integral > self._max_output:
            self._integral = self._max_output - self._proportional

        # Compute final output
        output = self._proportional + self._integral
        print(f"err: {err} Pterm: {self._proportional} Iterm: {self._integral}")
        return output

        




    