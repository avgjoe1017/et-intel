"""
Intelligence Brief Report Generator
Creates professional PDF reports with executive summary and visualizations
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
from .. import config

class IntelligenceBriefGenerator:
    """
    Generates professional PDF Intelligence Briefs
    """
    
    def __init__(self):
        self.reports_dir = config.REPORTS_DIR
        self.charts_dir = config.REPORTS_DIR / "charts"
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def generate_report(self, brief_data: dict, output_filename: str = None) -> Path:
        """
        Generate PDF report from intelligence brief data
        
        Args:
            brief_data: Dict from pipeline.generate_intelligence_brief()
            output_filename: Optional custom filename
        
        Returns:
            Path to generated PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ET_Intelligence_Brief_{timestamp}.pdf"
        
        output_path = self.reports_dir / output_filename
        
        # Create PDF
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(brief_data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(brief_data))
        story.append(PageBreak())
        
        # Velocity alerts (if any)
        if brief_data.get('velocity_alerts'):
            story.extend(self._create_velocity_section(brief_data))
            story.append(PageBreak())
        
        # Top entities section
        story.extend(self._create_entities_section(brief_data))
        story.append(PageBreak())
        
        # Sentiment analysis
        story.extend(self._create_sentiment_section(brief_data))
        story.append(PageBreak())
        
        # Storylines
        story.extend(self._create_storylines_section(brief_data))
        
        # Build PDF
        doc.build(story)
        
        print(f"\n✓ Intelligence Brief generated: {output_path}")
        return output_path
    
    def _create_title_page(self, data: dict) -> list:
        """Create title page"""
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=32,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        story = []
        
        # Add spacing
        story.append(Spacer(1, 2*inch))
        
        # Title
        story.append(Paragraph(config.REPORT_TITLE, title_style))
        story.append(Paragraph(config.REPORT_SUBTITLE, subtitle_style))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Date range
        date_range = data['metadata']['date_range']
        start = datetime.fromisoformat(date_range['start']).strftime('%B %d, %Y')
        end = datetime.fromisoformat(date_range['end']).strftime('%B %d, %Y')
        
        story.append(Paragraph(f"<b>Period:</b> {start} - {end}", subtitle_style))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            subtitle_style
        ))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Key metrics box
        metrics_data = [
            ['Total Comments Analyzed', f"{data['metadata']['total_comments']:,}"],
            ['Unique Posts', f"{data['metadata']['unique_posts']:,}"],
            ['Platforms', ', '.join(data['metadata']['platforms'])],
            ['Entities Tracked', f"{len(data['entities']['people']) + len(data['entities']['shows'])}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        
        story.append(metrics_table)
        
        return story
    
    def _create_executive_summary(self, data: dict) -> list:
        """Create executive summary section"""
        styles = getSampleStyleSheet()
        story = []
        
        # Section header
        story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Get top 5 entities by sentiment
        sentiment_summary = data['sentiment_summary']
        
        # Sort by mention count
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:5]
        
        # Key findings
        story.append(Paragraph("<b>KEY FINDINGS:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        findings = []
        
        # Top entity
        if sorted_entities:
            top_entity, top_data = sorted_entities[0]
            sentiment_label = self._sentiment_label(top_data['avg_sentiment'])
            findings.append(
                f"<b>{top_entity}</b> dominated conversation with {top_data['total_mentions']:,} mentions "
                f"({sentiment_label} sentiment, {top_data['avg_sentiment']:.2f})"
            )
        
        # Velocity alerts
        if data.get('velocity_alerts'):
            alert_count = len(data['velocity_alerts'])
            findings.append(
                f"<b>{alert_count} VELOCITY ALERTS</b> detected - significant sentiment shifts requiring attention"
            )
        
        # Top storyline
        if data['top_storylines']:
            top_storyline = data['top_storylines'][0]
            findings.append(
                f"<b>{top_storyline['storyline'].title()}</b> storyline active "
                f"({top_storyline['mention_count']} mentions, {top_storyline['percentage']:.1f}% of conversation)"
            )
        
        # Engagement patterns
        demo = data['demographics']
        if 'high_engagement_sentiment' in demo:
            findings.append(
                f"High-engagement comments skew <b>{self._sentiment_label(demo['high_engagement_sentiment'])}</b> "
                f"({demo['high_engagement_sentiment']:.2f}) vs. low-engagement ({demo['low_engagement_sentiment']:.2f})"
            )
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _create_velocity_section(self, data: dict) -> list:
        """Create velocity alerts section"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("VELOCITY ALERTS - RISK RADAR", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            "The following entities experienced significant sentiment velocity changes (±30% or more) "
            "within the past 72 hours:",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Create table
        table_data = [['Entity', 'Previous', 'Recent', 'Change', 'Status']]
        
        for alert in data['velocity_alerts'][:10]:  # Top 10 alerts
            status = "⚠ FALLING" if alert['sentiment_change'] < 0 else "↑ RISING"
            
            table_data.append([
                alert['entity'],
                f"{alert['previous_sentiment']:.2f}",
                f"{alert['recent_sentiment']:.2f}",
                f"{alert['percent_change']:+.1f}%",
                status
            ])
        
        t = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(t)
        
        return story
    
    def _create_entities_section(self, data: dict) -> list:
        """Create top entities section with chart"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("TOP ENTITIES BY VOLUME", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create chart
        chart_path = self._generate_entity_chart(data)
        if chart_path and chart_path.exists():
            img = RLImage(str(chart_path), width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        # Top 10 table
        sentiment_summary = data['sentiment_summary']
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:10]
        
        table_data = [['Rank', 'Entity', 'Mentions', 'Avg Sentiment', 'Dominant Emotion']]
        
        for i, (entity, entity_data) in enumerate(sorted_entities, 1):
            emotions = entity_data['emotion_breakdown']
            dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'N/A'
            
            table_data.append([
                str(i),
                entity[:30],
                f"{entity_data['total_mentions']:,}",
                f"{entity_data['avg_sentiment']:.2f}",
                dominant.title()
            ])
        
        t = Table(table_data, colWidths=[0.5*inch, 2*inch, 1*inch, 1*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        story.append(t)
        
        return story
    
    def _create_sentiment_section(self, data: dict) -> list:
        """Create sentiment breakdown section"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("EMOTION DISTRIBUTION", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create emotion chart
        chart_path = self._generate_emotion_chart(data)
        if chart_path and chart_path.exists():
            img = RLImage(str(chart_path), width=6*inch, height=3.5*inch)
            story.append(img)
        
        return story
    
    def _create_storylines_section(self, data: dict) -> list:
        """Create storylines section"""
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("ACTIVE STORYLINES", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        storylines = data['top_storylines'][:8]
        
        if storylines:
            table_data = [['Storyline', 'Mentions', '% of Total']]
            
            for sl in storylines:
                table_data.append([
                    sl['storyline'].title(),
                    f"{sl['mention_count']:,}",
                    f"{sl['percentage']:.1f}%"
                ])
            
            t = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 10)
            ]))
            
            story.append(t)
        else:
            story.append(Paragraph("No significant storylines detected in this period.", styles['Normal']))
        
        return story
    
    def _generate_entity_chart(self, data: dict) -> Path:
        """Generate bar chart of top entities"""
        sentiment_summary = data['sentiment_summary']
        sorted_entities = sorted(
            sentiment_summary.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:10]
        
        entities = [e[0][:25] for e in sorted_entities]
        mentions = [e[1]['total_mentions'] for e in sorted_entities]
        sentiments = [e[1]['avg_sentiment'] for e in sorted_entities]
        
        # Create color map based on sentiment
        colors_map = ['#2ecc71' if s > 0.3 else '#e74c3c' if s < -0.3 else '#95a5a6' for s in sentiments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(entities, mentions, color=colors_map)
        ax.set_xlabel('Total Mentions')
        ax.set_title('Top 10 Entities by Volume')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "entity_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _generate_emotion_chart(self, data: dict) -> Path:
        """Generate pie chart of emotion distribution"""
        # Aggregate all emotions
        all_emotions = {}
        for entity, entity_data in data['sentiment_summary'].items():
            for emotion, count in entity_data['emotion_breakdown'].items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + count
        
        emotions = list(all_emotions.keys())
        counts = list(all_emotions.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Emotion Distribution')
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "emotion_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"

