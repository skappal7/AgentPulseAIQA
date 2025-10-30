"""
CCRE Engine Module - Context Clustered Rule Engine
Enhanced with hierarchical category/subcategory classification
"""

import re
import ast
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class PIIRedactor:
    """Handles PII redaction using regex patterns"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'account_number': r'\b(?:account|acct|acc)[\s#:]*\d{6,}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'url': r'https?://[^\s]+',
        }
        
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.patterns.items()
        }
    
    def redact(self, text: str) -> str:
        """Redact all PII from text"""
        if not isinstance(text, str):
            return ""
        
        redacted = text
        for pii_type, pattern in self.compiled_patterns.items():
            redacted = pattern.sub(f'<REDACTED_{pii_type.upper()}>', redacted)
        
        return redacted


class HierarchicalCategoryMapper:
    """Maps categories to hierarchical parent-child relationships"""
    
    def __init__(self):
        # Define parent category mappings
        self.category_hierarchy = {
            # Account Related parent
            'account related': {
                'parent': 'Account Related',
                'subcategories': {
                    'dpa': 'Data Protection Access',
                    'verification': 'Verification Issue',
                    'verification issue': 'Verification Issue',
                    'email change': 'Email Change Request',
                    'password reset': 'Password Reset',
                    'activation': 'Account Activation',
                    'service protection': 'Service Protection',
                    'guarantee': 'Account Guarantee',
                }
            },
            
            # Billing parent
            'billing': {
                'parent': 'Billing',
                'subcategories': {
                    'waiver': 'Waiver / Fee',
                    'fee': 'Waiver / Fee',
                    'credit': 'Credit Request',
                    'refund': 'Refund Request',
                    'payment': 'Payment Issue',
                    'dispute': 'Disputed Charge',
                }
            },
            
            # Process parent
            'process': {
                'parent': 'Process',
                'subcategories': {
                    'verification': 'Verification Process',
                    'escalation': 'Escalation',
                    'follow-up': 'Follow-up Required',
                }
            },
            
            # DPA as standalone (can also be subcategory of Account Related)
            'dpa': {
                'parent': 'Account Related',  # Changed to be under Account Related
                'subcategories': {
                    'dpa': 'Data Protection Access',
                }
            },
            
            # Technical parent
            'technical': {
                'parent': 'Technical',
                'subcategories': {
                    'connectivity': 'Connectivity Issue',
                    'device': 'Device Issue',
                    'app': 'Application Issue',
                }
            },
        }
    
    def map_to_hierarchy(self, category: str, subcategory: str) -> Tuple[str, str]:
        """
        Map rule-based category/subcategory to hierarchical structure
        Returns: (parent_category, specific_subcategory)
        """
        category_lower = category.lower().strip()
        subcategory_lower = subcategory.lower().strip()
        
        # Find parent category
        parent_cat = category
        specific_subcat = subcategory
        
        # Check if category exists in hierarchy
        if category_lower in self.category_hierarchy:
            hierarchy = self.category_hierarchy[category_lower]
            parent_cat = hierarchy['parent']
            
            # Map subcategory if found
            if subcategory_lower in hierarchy['subcategories']:
                specific_subcat = hierarchy['subcategories'][subcategory_lower]
            else:
                # Use original subcategory if not in mapping
                specific_subcat = subcategory
        
        # Handle special case: If subcategory contains "dpa" but category isn't account related
        if 'dpa' in subcategory_lower:
            parent_cat = 'Account Related'
            specific_subcat = 'Data Protection Access'
        
        return parent_cat, specific_subcat


class CCRERuleEngine:
    """Context Clustered Rule Engine with Hierarchical Classification"""
    
    def __init__(self, rules_df: pd.DataFrame):
        self.rules_df = rules_df
        self.context_keywords = self._build_context_dictionaries()
        self.hierarchy_mapper = HierarchicalCategoryMapper()
        self._prepare_rules()
    
    def _build_context_dictionaries(self) -> Dict[str, List[str]]:
        """Build context keyword dictionaries for Stage 2 resolution"""
        return {
            'privacy': [
                'privacy policy', 'data protection', 'authorization', 
                'not authorized', 'security reason', 'gdpr', 'dpa',
                'personal data', 'consent', 'cannot provide',
                'cannot share', 'privacy reasons', 'confidential'
            ],
            'account': [
                'email', 'account', 'login', 'password', 'verify', 
                'verification', 'update email', 'reset password',
                'username', 'credentials', 'change email', 'modify account'
            ],
            'denial': [
                'unable', 'cannot', 'can not', "can't", 'failed', 
                'unsuccessful', 'system limitation', 'not possible',
                'restricted', 'blocked', 'denied'
            ],
            'empathy': [
                'sorry', 'apologize', 'apologies', 'inconvenience', 
                'thank you', 'appreciate', 'understand', 'patience',
                'frustrated', 'help you'
            ],
            'verification_intent': [
                'verify', 'verification', 'change email', 'update email',
                'change my email', 'update my email', 'email change',
                'modify account', 'update account', 'change account',
                'reset password', 'update password'
            ],
            'dpa_strong': [
                'data protection', 'privacy policy', 'gdpr', 
                'authorization required', 'not authorized', 'security reason',
                'cannot provide', 'cannot share', 'privacy reasons',
                'confidential', 'data privacy', 'personal information protection'
            ]
        }
    
    def _prepare_rules(self):
        """Parse required_groups and forbidden_terms from string format"""
        def safe_eval(val):
            if pd.isna(val) or val == '[]' or val == '':
                return []
            try:
                return ast.literal_eval(val)
            except:
                return []
        
        self.rules_df['required_groups_parsed'] = self.rules_df['required_groups'].apply(safe_eval)
        self.rules_df['forbidden_terms_parsed'] = self.rules_df['forbidden_terms'].apply(safe_eval)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.split()
    
    def _check_forbidden_terms(self, text: str, forbidden_terms: List[str]) -> bool:
        """Check if any forbidden terms present"""
        if not forbidden_terms:
            return False
        text_lower = text.lower()
        for term in forbidden_terms:
            if isinstance(term, str) and term.lower() in text_lower:
                return True
        return False
    
    def _calculate_proximity_bonus(self, text: str, matched_keywords: List[str]) -> float:
        """Calculate proximity bonus based on keyword distance"""
        if len(matched_keywords) < 2:
            return 0.0
        
        tokens = self._tokenize(text)
        positions = []
        
        for keyword in matched_keywords:
            keyword_lower = keyword.lower()
            for i, token in enumerate(tokens):
                if keyword_lower in token:
                    positions.append(i)
                    break
        
        if len(positions) < 2:
            return 0.0
        
        distances = [abs(positions[i] - positions[i-1]) for i in range(1, len(positions))]
        avg_distance = sum(distances) / len(distances)
        
        if avg_distance <= 5:
            return 0.2
        elif avg_distance <= 10:
            return 0.1
        elif avg_distance <= 20:
            return 0.05
        return 0.0
    
    def _detect_negations(self, text: str, matched_keywords: List[str]) -> float:
        """Detect negations near matched keywords"""
        negation_words = ['not', 'no', 'never', 'none', 'neither', "don't", "doesn't", 
                         "didn't", "won't", "wouldn't", "can't", "cannot"]
        
        tokens = self._tokenize(text)
        negation_count = 0
        
        for keyword in matched_keywords:
            keyword_lower = keyword.lower()
            for i, token in enumerate(tokens):
                if keyword_lower in token:
                    context_start = max(0, i - 3)
                    context_end = min(len(tokens), i + 4)
                    context_words = tokens[context_start:context_end]
                    
                    if any(neg in context_words for neg in negation_words):
                        negation_count += 1
                    break
        
        return min(0.6, negation_count * 0.3)
    
    def _score_rule(self, text: str, rule: pd.Series) -> Tuple[float, List[str], str]:
        """Score a single rule against text"""
        required_groups = rule['required_groups_parsed']
        forbidden_terms = rule['forbidden_terms_parsed']
        
        if self._check_forbidden_terms(text, forbidden_terms):
            return 0.0, [], "forbidden_term_present"
        
        matched_keywords = []
        groups_matched = 0
        text_lower = text.lower()
        
        for group in required_groups:
            if not isinstance(group, list):
                continue
            
            group_matched = False
            for keyword in group:
                if isinstance(keyword, str) and keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
                    group_matched = True
                    break
            
            if group_matched:
                groups_matched += 1
        
        if groups_matched == 0:
            return 0.0, [], "no_groups_matched"
        
        base_score = groups_matched / len(required_groups) if required_groups else 0.0
        proximity_bonus = self._calculate_proximity_bonus(text, matched_keywords)
        negation_penalty = self._detect_negations(text, matched_keywords)
        final_score = max(0.0, min(1.0, base_score + proximity_bonus - negation_penalty))
        
        reason = f"groups:{groups_matched}/{len(required_groups)}"
        if proximity_bonus > 0:
            reason += f"|prox:+{proximity_bonus:.2f}"
        if negation_penalty > 0:
            reason += f"|neg:-{negation_penalty:.2f}"
        
        return final_score, matched_keywords, reason
    
    def _stage1_activation(self, text: str) -> List[Dict[str, Any]]:
        """Stage 1: Activate all matching rules"""
        activated_rules = []
        
        for idx, rule in self.rules_df.iterrows():
            score, keywords, reason = self._score_rule(text, rule)
            
            if score >= 0.4:
                activated_rules.append({
                    'rule_id': rule['rule_id'],
                    'category': rule['category'],
                    'subcategory': rule['subcategory'],
                    'confidence': score,
                    'matched_keywords': keywords,
                    'activation_reason': reason
                })
        
        activated_rules.sort(key=lambda x: x['confidence'], reverse=True)
        return activated_rules
    
    def _detect_context(self, text: str) -> Dict[str, bool]:
        """Detect context types in text"""
        text_lower = text.lower()
        context_flags = {}
        
        for context_type, keywords in self.context_keywords.items():
            context_flags[context_type] = any(kw in text_lower for kw in keywords)
        
        return context_flags
    
    def _stage2_resolution(self, text: str, activated_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 2: Hierarchical contextual resolution
        Returns BOTH parent category and specific subcategory
        """
        if not activated_rules:
            return {
                'category': 'Unclassified',
                'subcategory': 'No Match',
                'confidence': 0.0,
                'resolve_reason': 'no_rules_activated',
                'matched_keywords': []
            }
        
        context = self._detect_context(text)
        text_lower = text.lower()
        
        # PRIORITY 1: Account Verification Intent (NOT DPA)
        has_verification_intent = context['verification_intent']
        has_dpa_strong = context['dpa_strong']
        
        if has_verification_intent and not has_dpa_strong:
            for rule in activated_rules:
                if 'dpa' not in rule['category'].lower() and 'dpa' not in rule['subcategory'].lower():
                    if ('account' in rule['category'].lower() or 
                        'verification' in rule['subcategory'].lower() or
                        'process' in rule['category'].lower()):
                        
                        # Map to hierarchy
                        parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
                            'Account Related', 'Verification Issue'
                        )
                        
                        return {
                            'category': parent_cat,
                            'subcategory': specific_subcat,
                            'confidence': rule['confidence'],
                            'resolve_reason': 'priority→account_verification_intent',
                            'matched_keywords': rule['matched_keywords']
                        }
        
        # PRIORITY 2: Strong DPA Context
        if has_dpa_strong or (context['privacy'] and not has_verification_intent):
            # This is actual DPA - map it as subcategory of Account Related
            parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
                'Account Related', 'DPA'
            )
            
            for rule in activated_rules:
                if 'dpa' in rule['category'].lower() or 'dpa' in rule['subcategory'].lower():
                    return {
                        'category': parent_cat,
                        'subcategory': specific_subcat,
                        'confidence': rule['confidence'],
                        'resolve_reason': 'priority→dpa_privacy_context',
                        'matched_keywords': rule['matched_keywords']
                    }
        
        # PRIORITY 3: Account + Denial → Verification Issue
        if context['account'] and context['denial']:
            for rule in activated_rules:
                if 'dpa' not in rule['category'].lower():
                    if 'account' in rule['category'].lower() or 'verification' in rule['subcategory'].lower():
                        parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
                            rule['category'], 'Verification Issue'
                        )
                        
                        return {
                            'category': parent_cat,
                            'subcategory': specific_subcat,
                            'confidence': rule['confidence'],
                            'resolve_reason': 'context→account+denial→verification',
                            'matched_keywords': rule['matched_keywords']
                        }
        
        # PRIORITY 4: Empathy Context
        if context['empathy'] and not context['denial']:
            for rule in activated_rules:
                if 'empathy' in rule['subcategory'].lower() or 'support' in rule['subcategory'].lower():
                    parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
                        rule['category'], rule['subcategory']
                    )
                    
                    return {
                        'category': parent_cat,
                        'subcategory': specific_subcat,
                        'confidence': rule['confidence'],
                        'resolve_reason': 'context→empathy+support',
                        'matched_keywords': rule['matched_keywords']
                    }
        
        # PRIORITY 5: Prefer non-DPA if DPA activated without strong indicators
        has_dpa_rule = any('dpa' in r['category'].lower() for r in activated_rules)
        if has_dpa_rule and not has_dpa_strong:
            for rule in activated_rules:
                if 'dpa' not in rule['category'].lower():
                    parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
                        rule['category'], rule['subcategory']
                    )
                    
                    return {
                        'category': parent_cat,
                        'subcategory': specific_subcat,
                        'confidence': rule['confidence'],
                        'resolve_reason': 'fallback→non_dpa_preferred',
                        'matched_keywords': rule['matched_keywords']
                    }
        
        # DEFAULT: Highest scoring rule with hierarchy mapping
        top_rule = activated_rules[0]
        parent_cat, specific_subcat = self.hierarchy_mapper.map_to_hierarchy(
            top_rule['category'], top_rule['subcategory']
        )
        
        return {
            'category': parent_cat,
            'subcategory': specific_subcat,
            'confidence': top_rule['confidence'],
            'resolve_reason': 'auto_resolved→highest_score',
            'matched_keywords': top_rule['matched_keywords']
        }
    
    def classify_transcript(self, transcript_id: Any, text: str) -> Dict[str, Any]:
        """Full CCRE classification pipeline with hierarchical output"""
        normalized_text = self._normalize_text(text)
        activated_rules = self._stage1_activation(normalized_text)
        result = self._stage2_resolution(normalized_text, activated_rules)
        
        result['transcript_id'] = transcript_id
        result['num_rules_activated'] = len(activated_rules)
        
        return result
