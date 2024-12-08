from flask import Flask, jsonify, request, send_file
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

class StableDiffusionProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = self.scheduler
        
        self.num_inference_steps = 100
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.timesteps = self.scheduler.timesteps
        
        self.current_latent = None
        self.initial_latent = None
        self.original_noise = None
        self.current_timestep_index = 0

    def process_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return preprocess(image).unsqueeze(0)

    def initialize_from_image(self, image: Image.Image):
        # image is now a PIL Image directly, not base64
        image_tensor = self.process_image(image)
        image_tensor = image_tensor.to(self.device, dtype=self.pipeline.unet.dtype)

        with torch.no_grad():
            self.current_latent = self.pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215
            self.current_latent = self.current_latent.detach()
            self.initial_latent = self.current_latent.clone()
            self.original_noise = torch.randn_like(self.current_latent)
            self.current_timestep_index = 0

    def add_noise(self, steps):
        max_steps = len(self.scheduler.timesteps)
        t_index = max(0, max_steps - steps)
        t = self.scheduler.timesteps[t_index]
        
        noisy_latent = self.scheduler.add_noise(self.current_latent, self.original_noise, t)
        
        with torch.no_grad():
            decoded = self.pipeline.vae.decode(noisy_latent / 0.18215).sample
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))
        
        self.current_latent = noisy_latent.detach()
        self.current_timestep_index = t_index
        return pil_image

    def denoise(self, steps, prompt_word="", modify_step=5, modify=False, guidance_scale=7.5):
        if steps < 1:
            steps = 1

        noisy_latent = self.current_latent.clone().detach()
        timesteps_to_use = self.scheduler.timesteps[self.current_timestep_index:self.current_timestep_index + steps]

        # Get conditional embeddings
        text_input = self.pipeline.tokenizer(prompt_word, padding="max_length", 
                                            max_length=self.pipeline.tokenizer.model_max_length,
                                            return_tensors="pt").input_ids.to(self.device)
        cond_embeddings = self.pipeline.text_encoder(text_input)[0]

        # Get unconditional embeddings
        uncond_input = self.pipeline.tokenizer([""], padding="max_length", 
                                                max_length=self.pipeline.tokenizer.model_max_length,
                                                return_tensors="pt").input_ids.to(self.device)
        uncond_embeddings = self.pipeline.text_encoder(uncond_input)[0]

        # Combine embeddings
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        with torch.no_grad():
            for i, t in enumerate(timesteps_to_use):
                # Unconditional pass
                noise_pred_uncond = self.pipeline.unet(noisy_latent, t, encoder_hidden_states=text_embeddings[:1]).sample
                # Conditional pass
                noise_pred_cond = self.pipeline.unet(noisy_latent, t, encoder_hidden_states=text_embeddings[1:]).sample

                # CFG guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Scheduler step
                step_output = self.pipeline.scheduler.step(noise_pred, t, noisy_latent)
                noisy_latent = step_output.prev_sample

                # Optional modification at a certain step
                if modify and i == modify_step - 1:
                    _, C, H, W = noisy_latent.shape
                    square_size = 10
                    x_start = W // 2 - square_size // 2
                    y_start = H // 2 - square_size // 2
                    noise = torch.randn_like(noisy_latent[:, :, y_start:y_start+square_size, x_start:x_start+square_size])
                    noisy_latent[:, :, y_start:y_start+square_size, x_start:x_start+square_size] += noise

            # Decode final latent
            decoded = self.pipeline.vae.decode(noisy_latent / 0.18215).sample
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

        self.current_latent = noisy_latent.detach()
        self.current_timestep_index += steps
        return pil_image


sd_processor = StableDiffusionProcessor()

@app.route('/', methods=['GET'])
def home():
    return "Hello from the root endpoint!"

# @app.route('/api/process', methods=['POST'], strict_slashes=False)
# def process_data():
#     data = request.json
#     # expected to receive {name:_, secret:_}
#     # Extract the name from the request body
#     name = data.get("name", "Guest")  # Default to "Guest" if name is not provided
#     secret = data.get("secret", "no secret")  # Default to "no secret" if secret is not provided

#     # Customize the response with the extracted name
#     result = {
#         "message": f"Hello, {name}! Your secret is {secret}."
#     }
#     return jsonify(result), 200

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Handles noise/denoise requests. 
    Accepts steps, type (noise/denoise), original_image, and prompt.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    # Extract parameters
    steps = data.get("steps")
    original_image = data.get("original_image")
    type = data.get("type")
    prompt = data.get("prompt")

    if not all([steps, original_image, type, prompt]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Call stable diffusion logic (stubbed here)
    try:
        processed_image = sd_processor.process_image(
            image=original_image, steps=steps, operation=type, prompt=prompt
        )
        return jsonify({"processed_image": processed_image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/greet/<username>', methods=['GET'])
def greet_user(username):
    # Return a JSON response
    return jsonify({
        "message": f"Hello, {username}! Welcome to the API."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


# posted on app.unaliu.com via tunneling