
"""
Context Compression Module
Advanced semantic and multimodal compression for efficient context management
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
import zlib
import base64
from datetime import datetime
from dataclasses import dataclass
import logging
import asyncio

# NLP imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Using basic text processing.")

logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """Result of compression operation"""
    compressed_data: Union[str, bytes, Dict[str, Any]]
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_method: str
    metadata: Dict[str, Any]
    timestamp: datetime

class ContextCompressionEngine:
    """Advanced context compression with semantic preservation"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        
        # Compression statistics
        self.compression_stats = {
            "total_operations": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "methods_used": {},
            "average_compression_ratio": 0.0
        }
        
        logger.info("Initialized ContextCompressionEngine")
    
    async def compress_context(self, context: Dict[str, Any], 
                              method: str = "adaptive", 
                              target_ratio: float = 0.7) -> CompressionResult:
        """Compress context data using specified method"""
        try:
            original_data = json.dumps(context, default=str)
            original_size = len(original_data.encode('utf-8'))
            
            # Choose compression method
            if method == "adaptive":
                compression_method = await self._choose_optimal_method(context)
            else:
                compression_method = method
            
            # Perform compression
            if compression_method == "semantic":
                result = await self._semantic_compression(context, target_ratio)
            elif compression_method == "statistical":
                result = await self._statistical_compression(context, target_ratio)
            elif compression_method == "hybrid":
                result = await self._hybrid_compression(context, target_ratio)
            elif compression_method == "multimodal":
                result = await self._multimodal_compression(context, target_ratio)
            else:
                result = await self._lossless_compression(context)
            
            # Calculate compression ratio
            compressed_size = self._calculate_size(result)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Update statistics
            await self._update_compression_stats(compression_method, original_size, compressed_size)
            
            return CompressionResult(
                compressed_data=result,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compression_method=compression_method,
                metadata={
                    "target_ratio": target_ratio,
                    "achieved_ratio": compression_ratio,
                    "quality_preserved": compression_ratio >= 0.8,
                    "semantic_loss": "minimal" if compression_method in ["semantic", "hybrid"] else "none"
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            raise
    
    async def decompress_context(self, compression_result: CompressionResult) -> Dict[str, Any]:
        """Decompress context data"""
        try:
            method = compression_result.compression_method
            compressed_data = compression_result.compressed_data
            
            if method == "semantic":
                return await self._semantic_decompression(compressed_data, compression_result.metadata)
            elif method == "statistical":
                return await self._statistical_decompression(compressed_data, compression_result.metadata)
            elif method == "hybrid":
                return await self._hybrid_decompression(compressed_data, compression_result.metadata)
            elif method == "multimodal":
                return await self._multimodal_decompression(compressed_data, compression_result.metadata)
            else:
                return await self._lossless_decompression(compressed_data)
            
        except Exception as e:
            logger.error(f"Error decompressing context: {e}")
            return {}
    
    async def _choose_optimal_method(self, context: Dict[str, Any]) -> str:
        """Choose optimal compression method based on context characteristics"""
        try:
            # Analyze context characteristics
            text_content = self._extract_text_content(context)
            has_multimodal = self._has_multimodal_content(context)
            
            text_length = len(text_content)
            complexity_score = self._calculate_complexity_score(context)
            
            # Decision logic
            if has_multimodal:
                return "multimodal"
            elif text_length > 5000 and complexity_score > 0.7:
                return "hybrid"
            elif text_length > 1000:
                return "semantic"
            elif complexity_score > 0.8:
                return "statistical"
            else:
                return "lossless"
                
        except Exception as e:
            logger.error(f"Error choosing compression method: {e}")
            return "lossless"
    
    async def _semantic_compression(self, context: Dict[str, Any], target_ratio: float) -> Dict[str, Any]:
        """Semantic compression with meaning preservation"""
        try:
            compressed_context = {}
            
            for key, value in context.items():
                if isinstance(value, str):
                    compressed_context[key] = await self._compress_text_semantic(value, target_ratio)
                elif isinstance(value, dict):
                    compressed_context[key] = await self._semantic_compression(value, target_ratio)
                elif isinstance(value, list):
                    compressed_context[key] = await self._compress_list_semantic(value, target_ratio)
                else:
                    compressed_context[key] = value
            
            return compressed_context
            
        except Exception as e:
            logger.error(f"Error in semantic compression: {e}")
            return context
    
    async def _compress_text_semantic(self, text: str, target_ratio: float) -> str:
        """Compress text while preserving semantic meaning"""
        try:
            if not NLTK_AVAILABLE or len(text) < 100:
                return text[:int(len(text) * target_ratio)]
            
            # Tokenize and process
            sentences = sent_tokenize(text)
            important_sentences = []
            
            for sentence in sentences:
                # Calculate sentence importance
                importance = self._calculate_sentence_importance(sentence, text)
                important_sentences.append((sentence, importance))
            
            # Sort by importance and take top sentences
            important_sentences.sort(key=lambda x: x[1], reverse=True)
            num_sentences = max(1, int(len(sentences) * target_ratio))
            selected_sentences = important_sentences[:num_sentences]
            
            # Reconstruct text maintaining order
            sentence_map = {sent: idx for idx, (sent, _) in enumerate(selected_sentences)}
            result_sentences = []
            
            for sentence in sentences:
                if sentence in sentence_map:
                    # Further compress the sentence
                    compressed_sentence = await self._compress_sentence(sentence)
                    result_sentences.append(compressed_sentence)
            
            return " ".join(result_sentences)
            
        except Exception as e:
            logger.error(f"Error compressing text semantically: {e}")
            return text[:int(len(text) * target_ratio)]
    
    async def _compress_sentence(self, sentence: str) -> str:
        """Compress individual sentence by removing non-essential words"""
        try:
            if not NLTK_AVAILABLE:
                words = sentence.split()
                return " ".join(words[i] for i in range(0, len(words), 2))  # Basic compression
            
            # Tokenize and POS tag
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Keep important words (nouns, verbs, adjectives, proper nouns)
            important_words = []
            for word, pos in pos_tags:
                if (pos.startswith(('NN', 'VB', 'JJ', 'NNP')) or 
                    word.lower() not in self.stop_words or 
                    len(word) > 8):  # Keep long words
                    
                    # Lemmatize word
                    lemmatized = self.lemmatizer.lemmatize(word.lower()) if self.lemmatizer else word.lower()
                    important_words.append(lemmatized)
            
            return " ".join(important_words)
            
        except Exception as e:
            logger.error(f"Error compressing sentence: {e}")
            return sentence
    
    def _calculate_sentence_importance(self, sentence: str, full_text: str) -> float:
        """Calculate sentence importance for semantic compression"""
        try:
            if not NLTK_AVAILABLE:
                return len(sentence) / len(full_text)  # Basic heuristic
            
            words = word_tokenize(sentence.lower())
            full_words = word_tokenize(full_text.lower())
            
            # Word frequency in document
            word_freq = {}
            for word in full_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate sentence score
            sentence_score = 0.0
            sentence_length = len(words)
            
            for word in words:
                if word not in self.stop_words:
                    # TF-IDF like scoring
                    tf = word_freq.get(word, 0) / len(full_words)
                    sentence_score += tf
            
            # Normalize by sentence length
            return sentence_score / sentence_length if sentence_length > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sentence importance: {e}")
            return 0.5
    
    async def _statistical_compression(self, context: Dict[str, Any], target_ratio: float) -> str:
        """Statistical compression using algorithms like zlib"""
        try:
            # Convert to JSON string
            json_str = json.dumps(context, separators=(',', ':'), default=str)
            
            # Apply zlib compression
            compressed_bytes = zlib.compress(json_str.encode('utf-8'))
            
            # Encode to base64 for JSON compatibility
            compressed_str = base64.b64encode(compressed_bytes).decode('utf-8')
            
            return compressed_str
            
        except Exception as e:
            logger.error(f"Error in statistical compression: {e}")
            return json.dumps(context, default=str)
    
    async def _hybrid_compression(self, context: Dict[str, Any], target_ratio: float) -> Dict[str, Any]:
        """Hybrid compression combining semantic and statistical methods"""
        try:
            # First apply semantic compression
            semantic_result = await self._semantic_compression(context, target_ratio * 1.2)
            
            # Then apply statistical compression to large text fields
            hybrid_result = {}
            
            for key, value in semantic_result.items():
                if isinstance(value, str) and len(value) > 500:
                    # Apply statistical compression to large strings
                    compressed_bytes = zlib.compress(value.encode('utf-8'))
                    compressed_str = base64.b64encode(compressed_bytes).decode('utf-8')
                    hybrid_result[key] = {
                        "_compressed": compressed_str,
                        "_type": "zlib_compressed_text"
                    }
                else:
                    hybrid_result[key] = value
            
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Error in hybrid compression: {e}")
            return context
    
    async def _multimodal_compression(self, context: Dict[str, Any], target_ratio: float) -> Dict[str, Any]:
        """Multimodal compression for mixed content types"""
        try:
            compressed_context = {}
            
            for key, value in context.items():
                if isinstance(value, str):
                    if self._is_image_data(value):
                        compressed_context[key] = await self._compress_image_data(value, target_ratio)
                    elif self._is_audio_data(value):
                        compressed_context[key] = await self._compress_audio_data(value, target_ratio)
                    else:
                        compressed_context[key] = await self._compress_text_semantic(value, target_ratio)
                elif isinstance(value, dict):
                    compressed_context[key] = await self._multimodal_compression(value, target_ratio)
                elif isinstance(value, list):
                    compressed_context[key] = await self._compress_list_semantic(value, target_ratio)
                else:
                    compressed_context[key] = value
            
            return compressed_context
            
        except Exception as e:
            logger.error(f"Error in multimodal compression: {e}")
            return context
    
    async def _lossless_compression(self, context: Dict[str, Any]) -> str:
        """Lossless compression for perfect reconstruction"""
        try:
            json_str = json.dumps(context, separators=(',', ':'), default=str)
            compressed_bytes = zlib.compress(json_str.encode('utf-8'))
            return base64.b64encode(compressed_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error in lossless compression: {e}")
            return json.dumps(context, default=str)
    
    async def _compress_list_semantic(self, items: List[Any], target_ratio: float) -> List[Any]:
        """Compress list by selecting most important items"""
        try:
            if len(items) <= 2:
                return items
            
            # Keep most important items based on size and complexity
            item_scores = []
            for idx, item in enumerate(items):
                score = self._calculate_item_importance(item)
                item_scores.append((idx, item, score))
            
            # Sort by importance and take top items
            item_scores.sort(key=lambda x: x[2], reverse=True)
            num_items = max(1, int(len(items) * target_ratio))
            
            # Maintain original order
            selected_indices = sorted([score[0] for score in item_scores[:num_items]])
            return [items[idx] for idx in selected_indices]
            
        except Exception as e:
            logger.error(f"Error compressing list: {e}")
            return items[:max(1, int(len(items) * target_ratio))]
    
    def _calculate_item_importance(self, item: Any) -> float:
        """Calculate importance score for list item"""
        try:
            if isinstance(item, dict):
                # More keys = higher importance
                return len(item) * 0.1
            elif isinstance(item, str):
                # Longer strings = higher importance (up to a point)
                return min(len(item) * 0.01, 1.0)
            elif isinstance(item, (int, float)):
                # Numbers have moderate importance
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.3
    
    # Helper methods for data type detection
    def _is_image_data(self, data: str) -> bool:
        """Check if string contains image data"""
        return data.startswith('data:image/') or data.startswith('/9j/') or len(data) > 1000 and data.isalnum()
    
    def _is_audio_data(self, data: str) -> bool:
        """Check if string contains audio data"""
        return data.startswith('data:audio/') or data.startswith('UklGR')
    
    async def _compress_image_data(self, image_data: str, target_ratio: float) -> Dict[str, Any]:
        """Compress image data (placeholder implementation)"""
        # In a real implementation, you would use PIL or similar
        return {
            "_compressed": image_data[:int(len(image_data) * target_ratio)],
            "_type": "compressed_image",
            "_original_length": len(image_data)
        }
    
    async def _compress_audio_data(self, audio_data: str, target_ratio: float) -> Dict[str, Any]:
        """Compress audio data (placeholder implementation)"""
        return {
            "_compressed": audio_data[:int(len(audio_data) * target_ratio)],
            "_type": "compressed_audio",
            "_original_length": len(audio_data)
        }
    
    # Decompression methods
    async def _semantic_decompression(self, compressed_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress semantically compressed data (best effort reconstruction)"""
        return compressed_data  # Semantic compression is lossy, return compressed version
    
    async def _statistical_decompression(self, compressed_data: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress statistically compressed data"""
        try:
            compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
            decompressed_str = zlib.decompress(compressed_bytes).decode('utf-8')
            return json.loads(decompressed_str)
            
        except Exception as e:
            logger.error(f"Error decompressing statistical data: {e}")
            return {}
    
    async def _hybrid_decompression(self, compressed_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress hybrid compressed data"""
        try:
            result = {}
            
            for key, value in compressed_data.items():
                if isinstance(value, dict) and value.get("_type") == "zlib_compressed_text":
                    # Decompress statistical part
                    compressed_bytes = base64.b64decode(value["_compressed"].encode('utf-8'))
                    decompressed_str = zlib.decompress(compressed_bytes).decode('utf-8')
                    result[key] = decompressed_str
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error decompressing hybrid data: {e}")
            return compressed_data
    
    async def _multimodal_decompression(self, compressed_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress multimodal data"""
        try:
            result = {}
            
            for key, value in compressed_data.items():
                if isinstance(value, dict) and "_type" in value:
                    if value["_type"] in ["compressed_image", "compressed_audio"]:
                        # For now, return compressed version (would implement proper decompression)
                        result[key] = value["_compressed"]
                    else:
                        result[key] = value
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error decompressing multimodal data: {e}")
            return compressed_data
    
    async def _lossless_decompression(self, compressed_data: str) -> Dict[str, Any]:
        """Decompress losslessly compressed data"""
        try:
            compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
            decompressed_str = zlib.decompress(compressed_bytes).decode('utf-8')
            return json.loads(decompressed_str)
            
        except Exception as e:
            logger.error(f"Error decompressing lossless data: {e}")
            return {}
    
    # Utility methods
    def _extract_text_content(self, context: Dict[str, Any]) -> str:
        """Extract all text content from context"""
        text_parts = []
        
        def extract_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(context)
        return " ".join(text_parts)
    
    def _has_multimodal_content(self, context: Dict[str, Any]) -> bool:
        """Check if context contains multimodal content"""
        text_content = self._extract_text_content(context)
        
        # Check for base64 encoded data or data URLs
        return ('data:image/' in text_content or 
                'data:audio/' in text_content or
                'data:video/' in text_content or
                any(len(str(v)) > 10000 and isinstance(v, str) for v in self._flatten_values(context)))
    
    def _flatten_values(self, obj) -> List[Any]:
        """Flatten all values from nested dict/list structure"""
        values = []
        
        if isinstance(obj, dict):
            for value in obj.values():
                values.extend(self._flatten_values(value))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(self._flatten_values(item))
        else:
            values.append(obj)
        
        return values
    
    def _calculate_complexity_score(self, context: Dict[str, Any]) -> float:
        """Calculate complexity score of context"""
        try:
            # Count nested levels, keys, and data types
            max_depth = self._calculate_max_depth(context)
            total_keys = self._count_total_keys(context)
            unique_types = len(set(type(v).__name__ for v in self._flatten_values(context)))
            
            # Normalize scores
            depth_score = min(max_depth / 5, 1.0)
            keys_score = min(total_keys / 50, 1.0)
            types_score = min(unique_types / 10, 1.0)
            
            return (depth_score + keys_score + types_score) / 3
            
        except Exception:
            return 0.5
    
    def _calculate_max_depth(self, obj, current_depth=0) -> int:
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            return max((self._calculate_max_depth(v, current_depth + 1) for v in obj.values()), default=current_depth)
        elif isinstance(obj, list):
            return max((self._calculate_max_depth(item, current_depth + 1) for item in obj), default=current_depth)
        else:
            return current_depth
    
    def _count_total_keys(self, obj) -> int:
        """Count total number of keys in nested structure"""
        if isinstance(obj, dict):
            return len(obj) + sum(self._count_total_keys(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._count_total_keys(item) for item in obj)
        else:
            return 0
    
    def _calculate_size(self, obj) -> int:
        """Calculate size of object in bytes"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            else:
                return len(json.dumps(obj, default=str).encode('utf-8'))
        except Exception:
            return 0
    
    async def _update_compression_stats(self, method: str, original_size: int, compressed_size: int):
        """Update compression statistics"""
        try:
            self.compression_stats["total_operations"] += 1
            self.compression_stats["total_original_bytes"] += original_size
            self.compression_stats["total_compressed_bytes"] += compressed_size
            
            if method not in self.compression_stats["methods_used"]:
                self.compression_stats["methods_used"][method] = 0
            self.compression_stats["methods_used"][method] += 1
            
            # Update average compression ratio
            if self.compression_stats["total_original_bytes"] > 0:
                self.compression_stats["average_compression_ratio"] = (
                    self.compression_stats["total_compressed_bytes"] / 
                    self.compression_stats["total_original_bytes"]
                )
            
        except Exception as e:
            logger.error(f"Error updating compression stats: {e}")
    
    async def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        return {
            "total_operations": self.compression_stats["total_operations"],
            "total_space_saved_mb": (
                (self.compression_stats["total_original_bytes"] - 
                 self.compression_stats["total_compressed_bytes"]) / (1024 * 1024)
            ),
            "average_compression_ratio": self.compression_stats["average_compression_ratio"],
            "methods_used": dict(self.compression_stats["methods_used"]),
            "efficiency_rating": self._calculate_efficiency_rating()
        }
    
    def _calculate_efficiency_rating(self) -> str:
        """Calculate overall efficiency rating"""
        ratio = self.compression_stats["average_compression_ratio"]
        if ratio < 0.5:
            return "excellent"
        elif ratio < 0.7:
            return "good"
        elif ratio < 0.9:
            return "moderate"
        else:
            return "minimal"
