import numpy as np

class PIDController:
    def __init__(self, model, sim, Kp=1.0, Ki=0.0, Kd=0.0, integral_clamp=10.0):
        self.model = model
        self.sim = sim
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_clamp = integral_clamp
        self.dt = model.opt.timestep
        self.pid_integral_error = None

    def pid_control_geared(self, target_qpos, qpos_marks):
        """
        Calculates the control signal for geared motors.
        It first computes the desired torque via PID, then converts this torque
        into a control signal based on the actuator's gear ratio.
        """
        assert target_qpos.shape[0] == self.model.nq, "Target qpos must match the number of qposes."
        if self.pid_integral_error is None:
            self.pid_integral_error = np.zeros(self.model.nq)
        target_qpos = target_qpos * qpos_marks
        current_qpos = self.sim.data.qpos * qpos_marks
        current_qvel = self.sim.data.qvel * qpos_marks
        position_error = target_qpos - current_qpos
        self.pid_integral_error += position_error * self.dt
        np.clip(self.pid_integral_error, -self.integral_clamp, self.integral_clamp, out=self.pid_integral_error)
        target_qvel = np.zeros_like(current_qvel)
        velocity_error = target_qvel - current_qvel
        
        joint_torques = (self.Kp * position_error + 
                         self.Ki * self.pid_integral_error + 
                         self.Kd * velocity_error)

        ctrl = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id >= 0:
                dof_address = self.model.jnt_dofadr[joint_id]
                desired_torque = joint_torques[dof_address]
                
                gear_ratio = self.model.actuator_gear[i, 0]
                if gear_ratio != 0:
                    unclamped_ctrl = desired_torque / gear_ratio
                else:
                    unclamped_ctrl = 0.0
                ctrlrange = self.model.actuator_ctrlrange[i]
                ctrl[i] = np.clip(unclamped_ctrl, ctrlrange[0], ctrlrange[1])
                
        return ctrl
    
    def reset(self):
        self.pid_integral_error = np.zeros(self.model.nq)

class PIDControllerSingle:
    def __init__(self, Kp, Ki, Kd, integral_clamp, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_clamp = integral_clamp
        self.setpoint = setpoint
        self._integral = 0
        self._previous_error = 0
        
    def reset(self):
        self._integral = 0
        self._previous_error = 0


    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_clamp, self.integral_clamp)
        derivative = (error - self._previous_error) / dt
        
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        
        self._previous_error = error
        
        return output

    def set_target(self, new_setpoint):
        self.setpoint = new_setpoint