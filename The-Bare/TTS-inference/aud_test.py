
import torch 
import random, os, sys
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

from dataclasses import dataclass
from pathlib import Path
import librosa
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# from chichat import perth # need chichat; pip install chichat
from chichat.chatterbox.models.t3 import T3
from chichat.chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chichat.chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chichat.chatterbox.models.tokenizers import EnTokenizer
from chichat.chatterbox.models.voice_encoder import VoiceEncoder
from chichat.chatterbox.models.t3.modules.cond_enc import T3Cond

REPO_ID = "callgg/chatterbox-encoder"

def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."
    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]
    # Remove multiple space chars
    text = " ".join(text.split())
    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text

try:
    from quant3 import convert_gguf_to_safetensors
except ImportError as e_top:
    try:
        from .quant3 import convert_gguf_to_safetensors
    except ImportError as e_rel:
        print(f"Warning: quant3.py not found or failed to import. GGUF conversion will fail. Errors: {e_top} | {e_rel}")
        # dummy fallback that does not crash immediately
        def convert_gguf_to_safetensors(*args, **kwargs):
            raise ImportError(f"quant3 module not found. Inner Error: {e_top}")

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # self.watermarker = perth.PerthImplicitWatermarker

    @classmethod
    def from_local(cls, ckpt_dir, device, vae_path, clip_path, model_path) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(vae_path)
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(clip_path)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(model_path), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device, vae_path, clip_path, model_path) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device, vae_path, clip_path, model_path)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]
            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
        return torch.from_numpy(wav).unsqueeze(0)

if __name__ == "__main__":
    import argparse
    import soundfile as sf
    import time
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS Inference (CLI)")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument("--ref_wav", type=str, help="Path to reference WAV file for cloning (optional)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    # Model arguments
    parser.add_argument("--vae_gguf", type=str, default="ve_fp32-f16.gguf", help="Path to VAE GGUF file")
    parser.add_argument("--clip_gguf", type=str, default="t3_cfg-q4_k_m.gguf", help="Path to CLIP/Encoder GGUF file")
    parser.add_argument("--model_gguf", type=str, default="s3gen-bf16.gguf", help="Path to S3Gen/Model GGUF file")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1. Check if GGUF files exist
    for f in [args.vae_gguf, args.clip_gguf, args.model_gguf]:
        if not os.path.exists(f):
            print(f"Error: GGUF file not found: {f}")
            print(f"Please ensure {f} is in the current directory or provide the correct path.")
            sys.exit(1)

    # 2. Convert GGUF to Safetensors
    print("Preparing models...")
    # Use cache dir instead of hardcoded relative path
    # If using absolute paths for inputs, place safetensors next to them
    
    use_bf16 = (DEVICE == "cuda")
    suffix = "bf16" if use_bf16 else "f32"
    
    def get_st_path(gguf_path):
        d = os.path.dirname(gguf_path)
        n = os.path.splitext(os.path.basename(gguf_path))[0]
        return os.path.join(d, f"{n}-{suffix}.safetensors")

    vae_s_path = get_st_path(args.vae_gguf)
    clip_s_path = get_st_path(args.clip_gguf)
    model_s_path = get_st_path(args.model_gguf)

    print(f"  VAE: {args.vae_gguf} -> {vae_s_path}")
    if not os.path.exists(vae_s_path):
        try:
            convert_gguf_to_safetensors(args.vae_gguf, vae_s_path, use_bf16)
        except Exception as e:
            print(f"Error converting VAE: {e}")
            sys.exit(1)
    else:
        print("  (Already converted)")

    print(f"  CLIP: {args.clip_gguf} -> {clip_s_path}")
    if not os.path.exists(clip_s_path):
        try:
            convert_gguf_to_safetensors(args.clip_gguf, clip_s_path, use_bf16)
        except Exception as e:
            print(f"Error converting CLIP: {e}")
            sys.exit(1)
    else:
        print("  (Already converted)")

    print(f"  Model: {args.model_gguf} -> {model_s_path}")
    if not os.path.exists(model_s_path):
        try:
            convert_gguf_to_safetensors(args.model_gguf, model_s_path, use_bf16)
        except Exception as e:
            print(f"Error converting Model: {e}")
            sys.exit(1)
    else:
        print("  (Already converted)")

    print("\nLoading model weights...")
    try:
        tts = ChatterboxTTS.from_pretrained(
            DEVICE, 
            vae_path=vae_s_path, 
            clip_path=clip_s_path, 
            model_path=model_s_path
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    if args.ref_wav:
        if os.path.exists(args.ref_wav):
            print(f"Using reference audio: {args.ref_wav}")
            tts.prepare_conditionals(args.ref_wav)
        else:
            print(f"Warning: Reference file {args.ref_wav} not found. Skipping.")
    elif tts.conds is None:
        print("Warning: No reference audio provided and no built-in conditionals found. Generation might fail or use defaults.")

    print(f"Generating audio for text: '{args.text}'")
    start = time.time()
    audio = tts.generate(args.text)
    end = time.time()
    
    print(f"Inference took {end - start:.2f}s")
    
    # Check output shape: (1, T) tensor -> needs (T,) numpy array for sf.write
    if hasattr(audio, 'squeeze'):
        audio = audio.squeeze().cpu().numpy()
        
    sf.write(args.output, audio, tts.sr)
    print(f"Audio saved to {args.output}")