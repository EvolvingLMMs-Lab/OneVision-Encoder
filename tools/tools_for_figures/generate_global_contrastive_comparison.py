#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an animated video to visualize the comparison between CLIP's contrastive learning
and the global contrastive learning approach using Manim.

Key differences:
1. CLIP: Image-Text pairs within a batch (limited negative samples)
2. Global: No text encoder, 1M concept centers from offline clustering as negatives

Usage:
    manim generate_global_contrastive_comparison.py -pql  # Preview low quality
    manim generate_global_contrastive_comparison.py -pqh  # Preview high quality
    manim generate_global_contrastive_comparison.py ComparisonVideo --format mp4 -qh
"""

import argparse
import manim as mn
import numpy as np


# Color definitions (converted from hex to Manim compatible)
mn.BLUE_600 = "#2563eb"
mn.BLUE_700 = "#1d4ed8"
mn.BLUE_400 = "#60a5fa"
SLATE_600 = "#475569"
SLATE_300 = "#cbd5e1"
SLATE_50 = "#f8fafc"
mn.BLUE_50 = "#eff6ff"

# Constants for consistency
DEFAULT_BATCH_SIZE = 8
NUM_POSITIVE_CENTERS = 10
NUM_VISIBLE_NEGATIVES = 25


def create_info_box(title_text, features, bg_color, title_color, position, width=5.5, height=4.0):
    """Helper function to create an info box with title and features.
    
    Args:
        title_text: Title text for the box
        features: List of feature strings
        bg_color: Background color
        title_color: Title text color
        position: Position of the box
        width: Width of the box
        height: Height of the box
        
    Returns:
        VGroup containing the box, title, and feature texts
    """
    box = mn.RoundedRectangle(
        corner_radius=0.2, width=width, height=height,
        fill_color=bg_color, fill_opacity=0.9,
        stroke_color=title_color, stroke_width=3
    )
    box.move_to(position)
    
    title = mn.Text(title_text, color=title_color, weight=mn.BOLD, font_size=28)
    title.move_to(box.get_top() + mn.DOWN * 0.5)
    
    feature_texts = mn.VGroup()
    for i, feature in enumerate(features):
        text = mn.Text(feature, color=SLATE_600, font_size=18)
        text.move_to(box.get_top() + mn.DOWN * (1.2 + i * 0.5))
        text.align_to(box.get_left() + mn.RIGHT * 0.3, mn.LEFT)
        feature_texts.add(text)
    
    return mn.VGroup(box, title, feature_texts)


class TitleScene(mn.Scene):
    """Create an introduction title frame with webpage-matching colors."""
    
    def construct(self):
        # Set background color to white
        self.camera.background_color = mn.WHITE
        
        # Title with blue gradient
        title = mn.Text("Cluster Discrimination Visualization", 
                    color=mn.BLUE_600, weight=mn.BOLD, font_size=48)
        title.move_to(mn.UP * 2.5)
        
        # Subtitle
        subtitle = mn.Text("CLIP vs. Global Contrastive Learning",
                       color=mn.BLUE_700, font_size=32)
        subtitle.next_to(title, mn.DOWN, buff=0.5)
        
        # Divider line
        divider = mn.Line(mn.LEFT * 5, mn.RIGHT * 5, color=SLATE_300, stroke_width=2)
        divider.next_to(subtitle, mn.DOWN, buff=0.5)
        
        # Content boxes using utility function
        clip_box = create_info_box(
            "CLIP",
            [
                "• Image-Text pairs",
                "• Batch-level contrastive",
                "• Limited negative samples",
                "  (batch size: 32-1024,",
                "   max ~32K negatives)",
                "• Dual encoders:",
                "  - Image Encoder",
                "  - Text Encoder",
                "• Cross-modal matching"
            ],
            SLATE_50,
            mn.BLUE_600,
            mn.LEFT * 3.5
        )
        
        global_box = create_info_box(
            "Global Contrastive",
            [
                "• Image only (no text)",
                "• Global negative sampling",
                "• 1M concept centers",
                "  (from offline clustering)",
                "• Single encoder:",
                "  - Image Encoder only",
                "• Sample negatives from",
                "  concept bank each batch"
            ],
            mn.BLUE_50,
            mn.BLUE_700,
            mn.RIGHT * 3.5
        )
        
        clip_box.next_to(divider, mn.DOWN, buff=1.0)
        global_box.next_to(divider, mn.DOWN, buff=1.0)
        
        # Add all elements
        self.play(
            mn.Write(title),
            mn.Write(subtitle),
            mn.Create(divider),
            run_time=1.5
        )
        self.play(
            mn.FadeIn(clip_box, shift=mn.LEFT * 0.5),
            mn.FadeIn(global_box, shift=mn.RIGHT * 0.5),
            run_time=1
        )
        self.wait(2)


class CLIPScene(mn.Scene):
    """Create a frame showing CLIP's contrastive learning."""
    
    def construct(self):
        self.camera.background_color = mn.WHITE
        batch_size = DEFAULT_BATCH_SIZE
        
        # Title
        title = mn.Text("CLIP: Batch-Level Image-Text Contrastive Learning",
                    color=mn.BLUE_600, font_size=32, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.5)
        self.play(mn.Write(title))
        
        # Batch info
        batch_info = mn.Text(f"Batch Size: {batch_size} pairs | Max ~32K negatives in large batches",
                         color=SLATE_600, font_size=20)
        batch_info.next_to(title, mn.DOWN, buff=0.3)
        self.play(mn.Write(batch_info))
        
        # Create image boxes on the left
        image_colors = [mn.RED, mn.GREEN, mn.BLUE, mn.YELLOW, mn.PURPLE, mn.TEAL, mn.ORANGE, mn.PINK]
        images = mn.VGroup()
        for i in range(batch_size):
            box = mn.Square(side_length=0.6, fill_color=image_colors[i], 
                        fill_opacity=0.7, stroke_color=mn.GREY)
            label = mn.Text(f"I{i+1}", color=mn.WHITE, font_size=16)
            label.move_to(box)
            img_group = mn.VGroup(box, label)
            img_group.move_to(mn.LEFT * 5.5 + mn.UP * (2 - i * 0.6))
            images.add(img_group)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(img) for img in images], lag_ratio=0.1))
        
        # Image Encoder
        img_encoder = mn.RoundedRectangle(
            corner_radius=0.1, width=1.2, height=4.5,
            fill_color="#283C50", fill_opacity=0.9,
            stroke_color="#6496C8", stroke_width=3
        )
        img_encoder.move_to(mn.LEFT * 3.5)
        img_enc_label = mn.Text("Image\nEncoder", color="#C8DCF0", font_size=18)
        img_enc_label.move_to(img_encoder)
        self.play(mn.FadeIn(img_encoder), mn.Write(img_enc_label))
        
        # Image embeddings
        img_embeddings = mn.VGroup()
        for i in range(batch_size):
            emb = mn.Dot(radius=0.15, color=image_colors[i])
            emb.move_to(mn.LEFT * 2 + mn.UP * (2 - i * 0.6))
            img_embeddings.add(emb)
        self.play(mn.LaggedStart(*[mn.GrowFromCenter(emb) for emb in img_embeddings], lag_ratio=0.1))
        
        # Text boxes on the right
        texts = mn.VGroup()
        for i in range(batch_size):
            box = mn.Square(side_length=0.6, fill_color=image_colors[i],
                        fill_opacity=0.7, stroke_color=mn.GREY)
            label = mn.Text(f"T{i+1}", color=mn.WHITE, font_size=16)
            label.move_to(box)
            text_group = mn.VGroup(box, label)
            text_group.move_to(mn.RIGHT * 5.5 + mn.UP * (2 - i * 0.6))
            texts.add(text_group)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(text) for text in texts], lag_ratio=0.1))
        
        # Text Encoder
        text_encoder = mn.RoundedRectangle(
            corner_radius=0.1, width=1.2, height=4.5,
            fill_color="#3C3228", fill_opacity=0.9,
            stroke_color="#FFB46E", stroke_width=3
        )
        text_encoder.move_to(mn.RIGHT * 3.5)
        text_enc_label = mn.Text("Text\nEncoder", color="#FFDCB4", font_size=18)
        text_enc_label.move_to(text_encoder)
        self.play(mn.FadeIn(text_encoder), mn.Write(text_enc_label))
        
        # Text embeddings
        text_embeddings = mn.VGroup()
        for i in range(batch_size):
            emb = mn.Dot(radius=0.15, color=image_colors[i])
            emb.move_to(mn.RIGHT * 2 + mn.UP * (2 - i * 0.6))
            text_embeddings.add(emb)
        self.play(mn.LaggedStart(*[mn.GrowFromCenter(emb) for emb in text_embeddings], lag_ratio=0.1))
        
        # Similarity Matrix in center
        matrix_label = mn.Text("Similarity Matrix", color=mn.GREY, font_size=20)
        matrix_label.move_to(mn.UP * 3.2)
        self.play(mn.Write(matrix_label))
        
        matrix = mn.VGroup()
        cell_size = 0.35
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    # Positive pair
                    cell = mn.Square(side_length=cell_size, fill_color=mn.GREEN,
                                fill_opacity=0.6, stroke_color=mn.GREY, stroke_width=1)
                else:
                    # Negative pair
                    cell = mn.Square(side_length=cell_size, fill_color=mn.RED,
                                fill_opacity=0.3, stroke_color=mn.GREY, stroke_width=1)
                cell.move_to(
                    (j - batch_size/2 + 0.5) * cell_size * mn.RIGHT +
                    (i - batch_size/2 + 0.5) * cell_size * mn.DOWN + mn.UP * 0.5
                )
                matrix.add(cell)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(cell) for cell in matrix], lag_ratio=0.02))
        
        # Animate highlighting different pairs
        for k in range(batch_size):
            animations = []
            for i in range(batch_size):
                for j in range(batch_size):
                    idx = i * batch_size + j
                    cell = matrix[idx]
                    if i == k and j == k:
                        animations.append(cell.animate.set_fill(color=mn.GREEN, opacity=0.9))
                    elif i == k or j == k:
                        animations.append(cell.animate.set_fill(opacity=0.5))
                    else:
                        animations.append(cell.animate.set_fill(opacity=0.2))
            
            self.play(*animations, run_time=0.4)
        
        # Reset matrix
        animations = []
        for i in range(batch_size):
            for j in range(batch_size):
                idx = i * batch_size + j
                cell = matrix[idx]
                if i == j:
                    animations.append(cell.animate.set_fill(color=mn.GREEN, opacity=0.6))
                else:
                    animations.append(cell.animate.set_fill(color=mn.RED, opacity=0.3))
        self.play(*animations, run_time=0.5)
        
        # Info box at bottom
        info_text = mn.VGroup(
            mn.Text(f"• Positive pairs: {batch_size} (diagonal elements)", 
                 color=SLATE_600, font_size=16),
            mn.Text(f"• Negative pairs: {batch_size * (batch_size - 1)} (off-diagonal)", 
                 color=SLATE_600, font_size=16),
            mn.Text(f"• Total comparisons: {batch_size * batch_size}", 
                 color=SLATE_600, font_size=16),
            mn.Text("• Challenge: Limited negatives scale with batch size", 
                 color=SLATE_600, font_size=16)
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.2)
        info_text.to_edge(mn.DOWN, buff=0.5)
        self.play(mn.FadeIn(info_text))
        
        self.wait(2)


class GlobalContrastiveScene(mn.Scene):
    """Create a frame showing Global Contrastive Learning with sampling animation."""
    
    def construct(self):
        self.camera.background_color = mn.WHITE
        batch_size = DEFAULT_BATCH_SIZE
        num_positive_centers = 10
        
        # Title
        title = mn.Text("Global Contrastive Learning: 1M Concept Centers",
                    color=mn.BLUE_700, font_size=32, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.5)
        self.play(mn.Write(title))
        
        # Batch info
        batch_info = mn.Text(f"Batch: {batch_size} images | Sampled Negatives: 1,024 | Total Concepts: 1,000,000",
                         color=SLATE_600, font_size=18)
        batch_info.next_to(title, mn.DOWN, buff=0.3)
        self.play(mn.Write(batch_info))
        
        # Create image boxes on the left
        image_colors = [mn.RED, mn.GREEN, mn.BLUE, mn.YELLOW, mn.PURPLE, mn.TEAL, mn.ORANGE, mn.PINK]
        images = mn.VGroup()
        for i in range(batch_size):
            box = mn.Square(side_length=0.5, fill_color=image_colors[i], 
                        fill_opacity=0.7, stroke_color=mn.GREY)
            label = mn.Text(f"I{i+1}", color=mn.WHITE, font_size=14)
            label.move_to(box)
            img_group = mn.VGroup(box, label)
            img_group.move_to(mn.LEFT * 6 + mn.UP * (2.5 - i * 0.5))
            images.add(img_group)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(img) for img in images], lag_ratio=0.1))
        
        # Image Encoder
        img_encoder = mn.RoundedRectangle(
            corner_radius=0.1, width=1.0, height=3.8,
            fill_color="#322D28", fill_opacity=0.9,
            stroke_color="#FFC878", stroke_width=3
        )
        img_encoder.move_to(mn.LEFT * 4.5)
        img_enc_label = mn.Text("Image\nEncoder", color="#FFE6C8", font_size=16)
        img_enc_label.move_to(img_encoder)
        self.play(mn.FadeIn(img_encoder), mn.Write(img_enc_label))
        
        # Image embeddings
        img_embeddings = mn.VGroup()
        for i in range(batch_size):
            emb = mn.Dot(radius=0.12, color=image_colors[i])
            emb.move_to(mn.LEFT * 3.2 + mn.UP * (2.5 - i * 0.5))
            img_embeddings.add(emb)
        self.play(mn.LaggedStart(*[mn.GrowFromCenter(emb) for emb in img_embeddings], lag_ratio=0.1))
        
        # Concept Bank visualization (right side)
        bank_border = mn.RoundedRectangle(
            corner_radius=0.15, width=8, height=6,
            fill_color="#191E23", fill_opacity=0.3,
            stroke_color="#C8A064", stroke_width=4
        )
        bank_border.move_to(mn.RIGHT * 2)
        
        bank_title = mn.Text("Concept Centers Bank", color="#FFDC8C", font_size=20, weight=mn.BOLD)
        bank_title.move_to(bank_border.get_top() + mn.DOWN * 0.4)
        bank_subtitle = mn.Text("(1,000,000 centers from offline clustering)", 
                            color="#C8B48C", font_size=14)
        bank_subtitle.next_to(bank_title, mn.DOWN, buff=0.2)
        
        self.play(mn.Create(bank_border), mn.Write(bank_title), mn.Write(bank_subtitle))
        
        # Create stable random positions for concept centers
        rng = np.random.default_rng(42)
        num_visible_concepts = 200
        concept_dots = mn.VGroup()
        
        for i in range(num_visible_concepts):
            # Random position within the bank
            x = bank_border.get_left()[0] + 0.5 + rng.random() * 7
            y = bank_border.get_bottom()[1] + 0.5 + rng.random() * 4.5
            
            # Different colors for different concept clusters
            cluster_id = i % 10
            colors = [
                "#FFB4B4", "#B4FFB4", "#B4B4FF", "#FFFFB4", "#FFB4FF",
                "#B4FFFF", "#DCC8B4", "#C8B4DC", "#B4DCC8", "#F0C8DC"
            ]
            
            dot = mn.Dot(radius=0.04, color=colors[cluster_id], fill_opacity=0.7)
            dot.move_to([x, y, 0])
            concept_dots.add(dot)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(dot) for dot in concept_dots], lag_ratio=0.005))
        
        # Animate sampling process for multiple samples
        for sample_idx in range(min(3, batch_size)):
            # Highlight current sample
            current_img = images[sample_idx]
            highlight = mn.SurroundingRectangle(current_img, color=mn.YELLOW, buff=0.1, stroke_width=4)
            self.play(mn.Create(highlight))
            
            # Select positive centers (clustered together)
            sample_seed = sample_idx * 1000
            pos_rng = np.random.default_rng(sample_seed)
            cluster_center_idx = pos_rng.integers(0, num_visible_concepts)
            
            # Find closest dots for positive centers
            center_pos = concept_dots[cluster_center_idx].get_center()
            distances = []
            for i, dot in enumerate(concept_dots):
                dist = np.linalg.norm(dot.get_center() - center_pos)
                distances.append((dist, i))
            distances.sort()
            positive_indices = [distances[i][1] for i in range(min(num_positive_centers, num_visible_concepts))]
            
            # Highlight positive centers in green
            positive_anims = []
            for idx in positive_indices:
                positive_anims.append(concept_dots[idx].animate.set_color(mn.GREEN).scale(2))
            self.play(*positive_anims, run_time=0.5)
            
            # Draw connection lines to positive centers
            lines = mn.VGroup()
            start_pos = img_embeddings[sample_idx].get_center()
            for idx in positive_indices[:5]:  # Show only a few lines to avoid clutter
                end_pos = concept_dots[idx].get_center()
                line = Dashedmn.Line(start_pos, end_pos, color=mn.GREEN, stroke_width=2, dash_length=0.1)
                lines.add(line)
            self.play(mn.Create(lines), run_time=0.5)
            
            # Select random negative centers (scattered)
            neg_rng = np.random.default_rng(sample_seed + 1)
            available = [i for i in range(num_visible_concepts) if i not in positive_indices]
            num_visible_negatives = 30
            negative_indices = neg_rng.choice(len(available), 
                                             size=min(num_visible_negatives, len(available)), 
                                             replace=False)
            negative_indices = [available[i] for i in negative_indices]
            
            # Highlight negative centers in orange/red
            negative_anims = []
            for idx in negative_indices:
                negative_anims.append(concept_dots[idx].animate.set_color(mn.ORANGE).scale(1.5))
            self.play(*negative_anims, run_time=0.5)
            
            self.wait(0.5)
            
            # Reset for next sample
            reset_anims = [mn.FadeOut(highlight), mn.FadeOut(lines)]
            for idx in positive_indices:
                reset_anims.append(concept_dots[idx].animate.set_color(mn.WHITE).scale(0.5))
            for idx in negative_indices:
                reset_anims.append(concept_dots[idx].animate.set_color(mn.WHITE).scale(1/1.5))
            self.play(*reset_anims, run_time=0.3)
        
        # Legend box at bottom
        legend = mn.VGroup(
            mn.VGroup(mn.Dot(radius=0.08, color=mn.YELLOW), mn.Text("Selected Sample", color=SLATE_600, font_size=14)).arrange(mn.RIGHT, buff=0.2),
            mn.VGroup(mn.Dot(radius=0.08, color=mn.GREEN), mn.Text(f"{num_positive_centers} Positive Centers (Clustered)", color=SLATE_600, font_size=14)).arrange(mn.RIGHT, buff=0.2),
            mn.VGroup(mn.Dot(radius=0.08, color=mn.ORANGE), mn.Text("Sampled Negatives (Scattered)", color=SLATE_600, font_size=14)).arrange(mn.RIGHT, buff=0.2),
            mn.VGroup(mn.Dot(radius=0.08, color=mn.WHITE), mn.Text("Other Concepts", color=SLATE_600, font_size=14)).arrange(mn.RIGHT, buff=0.2)
        ).arrange(mn.RIGHT, buff=0.5)
        legend.to_edge(mn.DOWN, buff=0.3)
        
        legend_box = mn.RoundedRectangle(
            corner_radius=0.1,
            width=legend.width + 0.5,
            height=legend.height + 0.3,
            fill_color="#141923", fill_opacity=0.3,
            stroke_color="#FFB46E", stroke_width=2
        )
        legend_box.move_to(legend)
        
        self.play(mn.FadeIn(legend_box), mn.FadeIn(legend))
        self.wait(2)


class ComparisonScene(mn.Scene):
    """Create a side-by-side comparison summary frame."""
    
    def construct(self):
        self.camera.background_color = mn.WHITE
        
        # Title
        title = mn.Text("Key Differences", color=mn.BLUE_600, font_size=44, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.5)
        self.play(mn.Write(title))
        
        # Divider
        divider = mn.Line(mn.UP * 2.5, mn.DOWN * 3, color=SLATE_300, stroke_width=3)
        self.play(mn.Create(divider))
        
        # CLIP side
        clip_title = mn.Text("CLIP Approach", color=mn.BLUE_600, font_size=28, weight=mn.BOLD)
        clip_title.move_to(mn.LEFT * 3.5 + mn.UP * 2)
        
        clip_points = [
            ("Architecture", "Dual encoders (Image + Text)"),
            ("Input", "Image-Text pairs"),
            ("Negatives", "Within batch (32-1024 pairs)"),
            ("Limitation", "Limited by batch size"),
            ("Benefit", "Cross-modal alignment"),
            ("Scale", "~400M pairs training"),
        ]
        
        clip_items = mn.VGroup()
        for i, (label, value) in enumerate(clip_points):
            label_text = mn.Text(label + ":", color=SLATE_600, font_size=18, weight=mn.BOLD)
            value_box = mn.RoundedRectangle(
                corner_radius=0.08, width=5, height=0.5,
                fill_color=SLATE_50, fill_opacity=0.8,
                stroke_color=mn.BLUE_400, stroke_width=2
            )
            value_text = mn.Text(value, color="#1E293B", font_size=16)
            value_text.move_to(value_box)
            
            item = mn.VGroup(label_text, mn.VGroup(value_box, value_text)).arrange(mn.DOWN, buff=0.15, aligned_edge=mn.LEFT)
            item.move_to(mn.LEFT * 3.5 + mn.UP * (1 - i * 0.9))
            clip_items.add(item)
        
        # Global side
        global_title = mn.Text("Global Contrastive", color=mn.BLUE_700, font_size=28, weight=mn.BOLD)
        global_title.move_to(mn.RIGHT * 3.5 + mn.UP * 2)
        
        global_points = [
            ("Architecture", "Single encoder (Image only)"),
            ("Input", "Images only"),
            ("Negatives", "Sampled from 1M concept centers"),
            ("Limitation", "No text alignment"),
            ("Benefit", "Massive negative pool"),
            ("Scale", "~1M concept centers"),
        ]
        
        global_items = mn.VGroup()
        for i, (label, value) in enumerate(global_points):
            label_text = mn.Text(label + ":", color=SLATE_600, font_size=18, weight=mn.BOLD)
            value_box = mn.RoundedRectangle(
                corner_radius=0.08, width=5, height=0.5,
                fill_color=mn.BLUE_50, fill_opacity=0.8,
                stroke_color=mn.BLUE_400, stroke_width=2
            )
            value_text = mn.Text(value, color="#1E3A5F", font_size=16)
            value_text.move_to(value_box)
            
            item = mn.VGroup(label_text, mn.VGroup(value_box, value_text)).arrange(mn.DOWN, buff=0.15, aligned_edge=mn.LEFT)
            item.move_to(mn.RIGHT * 3.5 + mn.UP * (1 - i * 0.9))
            global_items.add(item)
        
        # Animate everything
        self.play(mn.Write(clip_title), mn.Write(global_title))
        self.play(
            mn.LaggedStart(*[mn.FadeIn(item, shift=mn.DOWN * 0.2) for item in clip_items], lag_ratio=0.2),
            mn.LaggedStart(*[mn.FadeIn(item, shift=mn.DOWN * 0.2) for item in global_items], lag_ratio=0.2),
            run_time=3
        )
        self.wait(3)


class ComparisonVideo(mn.Scene):
    """Main scene that combines all scenes into one video."""
    
    def construct(self):
        # Set white background
        self.camera.background_color = mn.WHITE
        
        # Scene 1: Title Scene
        self._render_title_scene()
        self.clear()
        self.wait(0.5)
        
        # Scene 2: CLIP Scene
        self._render_clip_scene()
        self.clear()
        self.wait(0.5)
        
        # Scene 3: Global Contrastive Scene
        self._render_global_scene()
        self.clear()
        self.wait(0.5)
        
        # Scene 4: Comparison Scene
        self._render_comparison_scene()
    
    def _render_title_scene(self):
        """Render title scene content."""
        # Title
        title = mn.Text("Cluster Discrimination Visualization", 
                    color=mn.BLUE_600, weight=mn.BOLD, font_size=48)
        title.move_to(mn.UP * 2.5)
        
        subtitle = mn.Text("CLIP vs. Global Contrastive Learning",
                       color=mn.BLUE_700, font_size=32)
        subtitle.next_to(title, mn.DOWN, buff=0.5)
        
        divider = mn.Line(mn.LEFT * 5, mn.RIGHT * 5, color=SLATE_300, stroke_width=2)
        divider.next_to(subtitle, mn.DOWN, buff=0.5)
        
        # Create info boxes using helper
        clip_box = self._create_info_box(
            "CLIP",
            [
                "• Image-Text pairs",
                "• Batch-level contrastive",
                "• Limited negative samples",
                "• Dual encoders",
                "• Cross-modal matching"
            ],
            SLATE_50,
            mn.BLUE_600,
            mn.LEFT * 3.5
        )
        
        global_box = self._create_info_box(
            "Global Contrastive",
            [
                "• Image only (no text)",
                "• Global negative sampling",
                "• 1M concept centers",
                "• Single encoder",
                "• Sample from concept bank"
            ],
            mn.BLUE_50,
            mn.BLUE_700,
            mn.RIGHT * 3.5
        )
        
        clip_box.next_to(divider, mn.DOWN, buff=1.0)
        global_box.next_to(divider, mn.DOWN, buff=1.0)
        
        self.play(mn.Write(title), mn.Write(subtitle), mn.Create(divider), run_time=1.5)
        self.play(mn.FadeIn(clip_box, shift=mn.LEFT * 0.5), mn.FadeIn(global_box, shift=mn.RIGHT * 0.5), run_time=1)
        self.wait(2)
    
    def _create_info_box(self, title_text, features, bg_color, title_color, position):
        """Helper to create an info box."""
        box = mn.RoundedRectangle(
            corner_radius=0.2, width=5.5, height=4.0,
            fill_color=bg_color, fill_opacity=0.9,
            stroke_color=title_color, stroke_width=3
        )
        box.move_to(position)
        
        title = mn.Text(title_text, color=title_color, weight=mn.BOLD, font_size=28)
        title.move_to(box.get_top() + mn.DOWN * 0.5)
        
        feature_texts = mn.VGroup()
        for i, feature in enumerate(features):
            text = mn.Text(feature, color=SLATE_600, font_size=18)
            text.move_to(box.get_top() + mn.DOWN * (1.2 + i * 0.5))
            text.align_to(box.get_left() + mn.RIGHT * 0.3, mn.LEFT)
            feature_texts.add(text)
        
        return mn.VGroup(box, title, feature_texts)
    
    def _render_clip_scene(self):
        """Render CLIP scene content."""
        batch_size = DEFAULT_BATCH_SIZE
        
        title = mn.Text("CLIP: Batch-Level Image-Text Contrastive Learning",
                    color=mn.BLUE_600, font_size=28, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.3)
        self.play(mn.Write(title), run_time=0.8)
        
        # Simplified CLIP visualization
        image_colors = [mn.RED, mn.GREEN, mn.BLUE, mn.YELLOW, mn.PURPLE, mn.TEAL, mn.ORANGE, mn.PINK]
        images = mn.VGroup()
        for i in range(batch_size):
            box = mn.Square(side_length=0.5, fill_color=image_colors[i], 
                        fill_opacity=0.7, stroke_color=mn.GREY)
            label = mn.Text(f"I{i+1}", color=mn.WHITE, font_size=14)
            label.move_to(box)
            img_group = mn.VGroup(box, label)
            img_group.move_to(mn.LEFT * 5.5 + mn.UP * (2 - i * 0.5))
            images.add(img_group)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(img) for img in images], lag_ratio=0.05), run_time=1.5)
        
        # Show similarity matrix
        matrix_label = mn.Text("Similarity Matrix", color=mn.GREY, font_size=18)
        matrix_label.move_to(mn.UP * 2.8)
        self.play(mn.Write(matrix_label), run_time=0.5)
        
        matrix = mn.VGroup()
        cell_size = 0.3
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    cell = mn.Square(side_length=cell_size, fill_color=mn.GREEN,
                                fill_opacity=0.6, stroke_color=mn.GREY, stroke_width=1)
                else:
                    cell = mn.Square(side_length=cell_size, fill_color=mn.RED,
                                fill_opacity=0.3, stroke_color=mn.GREY, stroke_width=1)
                cell.move_to(
                    (j - batch_size/2 + 0.5) * cell_size * mn.RIGHT +
                    (i - batch_size/2 + 0.5) * cell_size * mn.DOWN + mn.UP * 0.3
                )
                matrix.add(cell)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(cell) for cell in matrix], lag_ratio=0.01), run_time=1.5)
        
        # Animate a few highlights
        for k in [0, 3, 7]:
            animations = []
            for i in range(batch_size):
                for j in range(batch_size):
                    idx = i * batch_size + j
                    cell = matrix[idx]
                    if i == k and j == k:
                        animations.append(cell.animate.set_fill(color=mn.GREEN, opacity=0.9))
                    elif i == k or j == k:
                        animations.append(cell.animate.set_fill(opacity=0.5))
                    else:
                        animations.append(cell.animate.set_fill(opacity=0.2))
            self.play(*animations, run_time=0.3)
        
        # Info text
        info_text = mn.Text(f"Limited to {batch_size}x{batch_size} = {batch_size*batch_size} comparisons per batch", 
                        color=SLATE_600, font_size=16)
        info_text.to_edge(mn.DOWN, buff=0.5)
        self.play(mn.FadeIn(info_text), run_time=0.5)
        self.wait(2)
    
    def _render_global_scene(self):
        """Render global contrastive scene content."""
        batch_size = 6  # Reduced for clearer visualization
        
        title = mn.Text("Global Contrastive Learning: 1M Concept Centers",
                    color=mn.BLUE_700, font_size=28, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.3)
        self.play(mn.Write(title), run_time=0.8)
        
        # Images on left
        image_colors = [mn.RED, mn.GREEN, mn.BLUE, mn.YELLOW, mn.PURPLE, mn.TEAL]
        images = mn.VGroup()
        for i in range(batch_size):
            box = mn.Square(side_length=0.4, fill_color=image_colors[i], 
                        fill_opacity=0.7, stroke_color=mn.GREY)
            label = mn.Text(f"I{i+1}", color=mn.WHITE, font_size=12)
            label.move_to(box)
            img_group = mn.VGroup(box, label)
            img_group.move_to(mn.LEFT * 6 + mn.UP * (2 - i * 0.6))
            images.add(img_group)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(img) for img in images], lag_ratio=0.05), run_time=1)
        
        # Concept bank
        bank_border = mn.RoundedRectangle(
            corner_radius=0.15, width=9, height=5.5,
            fill_color="#191E23", fill_opacity=0.2,
            stroke_color="#C8A064", stroke_width=4
        )
        bank_border.move_to(mn.RIGHT * 1.5)
        
        bank_title = mn.Text("Concept Centers Bank (1M centers)", 
                         color="#FFDC8C", font_size=18, weight=mn.BOLD)
        bank_title.move_to(bank_border.get_top() + mn.DOWN * 0.4)
        
        self.play(mn.Create(bank_border), mn.Write(bank_title), run_time=1)
        
        # Create concept dots
        rng = np.random.default_rng(42)
        num_visible_concepts = 150
        concept_dots = mn.VGroup()
        
        for i in range(num_visible_concepts):
            x = bank_border.get_left()[0] + 0.5 + rng.random() * 8
            y = bank_border.get_bottom()[1] + 0.5 + rng.random() * 4.5
            
            dot = mn.Dot(radius=0.04, color=mn.WHITE, fill_opacity=0.5)
            dot.move_to([x, y, 0])
            concept_dots.add(dot)
        
        self.play(mn.LaggedStart(*[mn.FadeIn(dot) for dot in concept_dots], lag_ratio=0.003), run_time=1.5)
        
        # Demonstrate sampling for one image
        highlight = mn.SurroundingRectangle(images[2], color=mn.YELLOW, buff=0.1, stroke_width=4)
        self.play(mn.Create(highlight), run_time=0.5)
        
        # Highlight some positive centers (green)
        positive_anims = []
        for i in range(10):
            positive_anims.append(concept_dots[20+i*3].animate.set_color(mn.GREEN).scale(2))
        self.play(*positive_anims, run_time=0.7)
        
        # Highlight some negative centers (orange)
        negative_anims = []
        for i in range(25):
            negative_anims.append(concept_dots[i*6].animate.set_color(mn.ORANGE).scale(1.5))
        self.play(*negative_anims, run_time=0.7)
        
        self.wait(2)
    
    def _render_comparison_scene(self):
        """Render comparison scene content."""
        title = mn.Text("Key Differences", color=mn.BLUE_600, font_size=40, weight=mn.BOLD)
        title.to_edge(mn.UP, buff=0.5)
        self.play(mn.Write(title), run_time=0.8)
        
        divider = mn.Line(mn.UP * 2.5, mn.DOWN * 3, color=SLATE_300, stroke_width=3)
        self.play(mn.Create(divider), run_time=0.5)
        
        # CLIP side
        clip_title = mn.Text("CLIP", color=mn.BLUE_600, font_size=32, weight=mn.BOLD)
        clip_title.move_to(mn.LEFT * 3.5 + mn.UP * 2.2)
        
        clip_items = mn.VGroup(
            mn.Text("• Dual encoders", color=SLATE_600, font_size=18),
            mn.Text("• Image-Text pairs", color=SLATE_600, font_size=18),
            mn.Text("• Batch-limited negatives", color=SLATE_600, font_size=18),
            mn.Text("• Cross-modal alignment", color=SLATE_600, font_size=18),
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.4)
        clip_items.move_to(mn.LEFT * 3.5 + mn.DOWN * 0.2)
        
        # Global side
        global_title = mn.Text("Global Contrastive", color=mn.BLUE_700, font_size=32, weight=mn.BOLD)
        global_title.move_to(mn.RIGHT * 3.5 + mn.UP * 2.2)
        
        global_items = mn.VGroup(
            mn.Text("• Single encoder", color=SLATE_600, font_size=18),
            mn.Text("• Images only", color=SLATE_600, font_size=18),
            mn.Text("• 1M concept centers", color=SLATE_600, font_size=18),
            mn.Text("• Massive negative pool", color=SLATE_600, font_size=18),
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.4)
        global_items.move_to(mn.RIGHT * 3.5 + mn.DOWN * 0.2)
        
        self.play(mn.Write(clip_title), mn.Write(global_title), run_time=0.8)
        self.play(
            mn.LaggedStart(*[mn.FadeIn(item, shift=mn.DOWN * 0.2) for item in clip_items], lag_ratio=0.2),
            mn.LaggedStart(*[mn.FadeIn(item, shift=mn.DOWN * 0.2) for item in global_items], lag_ratio=0.2),
            run_time=2
        )
        self.wait(3)


if __name__ == "__main__":
    # This script is meant to be run with manim command
    # Example: manim generate_global_contrastive_comparison.py ComparisonVideo -pql
    print("Please run this script with manim command:")
    print("  manim generate_global_contrastive_comparison.py ComparisonVideo -pql  # Preview low quality")
    print("  manim generate_global_contrastive_comparison.py ComparisonVideo -pqh  # Preview high quality")
    print("  manim generate_global_contrastive_comparison.py ComparisonVideo --format mp4 -qh  # Render high quality MP4")
