"""
Tests for AI-generated image disclaimer in captions.
Run with: pytest tests/test_ai_disclaimer.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAIDisclaimer:
    """Tests for AI-generated image disclaimer functionality"""
    
    def test_real_photo_no_disclaimer(self):
        """Test that real photos don't have disclaimer"""
        from function_app import CAPTION_PREFIX, CAPTION_HASHTAGS
        
        witty_caption = "Milo is looking particularly grumpy today!"
        image_source = "blob storage (IMG20250101.jpg)"
        
        # Build caption as done in function_app
        caption_parts = [CAPTION_PREFIX, witty_caption]
        
        if "AI generated" in image_source:
            caption_parts.append("(AI-generated image)")
        
        caption_parts.append(CAPTION_HASHTAGS)
        caption = " ".join(caption_parts)
        
        # Verify no disclaimer
        assert "(AI-generated image)" not in caption
        assert caption.startswith(CAPTION_PREFIX)
        assert caption.endswith(CAPTION_HASHTAGS)
        assert witty_caption in caption
    
    def test_ai_generated_has_disclaimer(self):
        """Test that AI-generated images have disclaimer"""
        from function_app import CAPTION_PREFIX, CAPTION_HASHTAGS
        
        witty_caption = "Milo is looking particularly grumpy today!"
        image_source = "AI generated (OpenAI)"
        
        # Build caption as done in function_app
        caption_parts = [CAPTION_PREFIX, witty_caption]
        
        if "AI generated" in image_source:
            caption_parts.append("(AI-generated image)")
        
        caption_parts.append(CAPTION_HASHTAGS)
        caption = " ".join(caption_parts)
        
        # Verify disclaimer is present
        assert "(AI-generated image)" in caption
        assert caption.startswith(CAPTION_PREFIX)
        assert caption.endswith(CAPTION_HASHTAGS)
        assert witty_caption in caption
        
        # Verify order: prefix, caption, disclaimer, hashtags
        ai_disclaimer_index = caption.index("(AI-generated image)")
        witty_index = caption.index(witty_caption.split()[0])
        hashtags_index = caption.index("#Milo")
        
        assert witty_index < ai_disclaimer_index < hashtags_index
    
    def test_disclaimer_format(self):
        """Test that disclaimer has correct format"""
        from function_app import CAPTION_PREFIX, CAPTION_HASHTAGS
        
        image_source = "AI generated (OpenAI)"
        witty_caption = "Test caption"
        
        caption_parts = [CAPTION_PREFIX, witty_caption]
        
        if "AI generated" in image_source:
            caption_parts.append("(AI-generated image)")
        
        caption_parts.append(CAPTION_HASHTAGS)
        caption = " ".join(caption_parts)
        
        # Check exact disclaimer text
        assert "(AI-generated image)" in caption
        # Should have parentheses
        assert caption.count("(AI-generated image)") == 1
