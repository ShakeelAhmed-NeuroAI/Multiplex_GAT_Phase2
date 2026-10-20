import torch
import numpy as np
from multiplex_model import MultiplexBrainEngine

class FeedbackController:
    def __init__(self, target=0.1500, kp=0.4, ki=0.1, kd=0.05, safety_limit=0.1000):
        self.ref_target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.slew_rate_limit = safety_limit
        
        # State registers
        self.error_sum = 0.0
        self.last_error = 0.0

    def compute_next_step(self, measured_val, current_u):
        error = self.ref_target - float(measured_val)
        
        # Anti-windup clamping integration bounds
        self.error_sum = max(-2.0, min(2.0, self.error_sum + error))
        error_diff = error - self.last_error
        self.last_error = error
        
        u_target = (self.kp * error) + (self.ki * self.error_sum) + (self.kd * error_diff)
        
        # Enforce step constraint limits
        u_delta = u_target - current_u
        if abs(u_delta) > self.slew_rate_limit:
            u_out = current_u + np.sign(u_delta) * self.slew_rate_limit
            is_clamped = True
        else:
            u_out = u_target
            is_clamped = False
            
        return u_out, error, is_clamped

def execute_simulation_run():
    print(">>> Firing up closed-loop tracking loop execution check...")
    
    model = MultiplexBrainEngine()
    model.eval()
    
    loop = FeedbackController()
    u_state = 0.0
    
    print("\n--- Telemetry Updates ---")
    for step in range(5):
        # Simulate active array ingestion frames
        feat_in = torch.randn(1, 32, 5)
        adj_in = torch.rand(1, 3, 32, 32)
        
        with torch.no_grad():
            score = model(feat_in, adj_in)
            
        u_state, current_err, clamped_flag = loop.compute_next_step(score.item(), u_state)
        print(f"Step [{step+1}/5] | Pred: {score.item():.4f} | Err: {current_err:.4f} | Output U: {u_state:.4f} | Safety Clamp: {clamped_flag}")

if __name__ == "__main__":
    execute_simulation_run()
