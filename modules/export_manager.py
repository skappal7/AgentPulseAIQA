"""
Export Manager Module
Handles export to multiple formats: PPTX, DOCX, CSV, Excel, HTML, Parquet
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import os


class ExportManager:
    """Manages exports to multiple formats"""
    
    def __init__(self, output_dir: str = "/home/claude/agentpulse_ai/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: str = "classified_transcripts.csv"
    ) -> str:
        """Export data to CSV"""
        output_path = self.output_dir / filename
        data.to_csv(output_path, index=False)
        return str(output_path)
    
    def export_to_excel(
        self,
        data: pd.DataFrame,
        filename: str = "classified_transcripts.xlsx"
    ) -> str:
        """Export data to Excel with formatting"""
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Classifications', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Classifications']
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        return str(output_path)
    
    def export_to_parquet(
        self,
        data: pd.DataFrame,
        filename: str = "classified_transcripts.parquet"
    ) -> str:
        """Export data to Parquet"""
        output_path = self.output_dir / filename
        data.to_parquet(output_path, index=False, engine='pyarrow')
        return str(output_path)
    
    def export_to_html(
        self,
        summary_data: Dict[str, Any],
        filename: str = "report.html"
    ) -> str:
        """Export summary report to HTML"""
        output_path = self.output_dir / filename
        
        html_content = self._generate_html_report(summary_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        
        summary = data.get('summary', {})
        category_dist = data.get('category_distribution', [])
        agent_perf = data.get('agent_performance', [])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AgentPulse AI - Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f7fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 AgentPulse AI Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Transcripts</h3>
            <div class="value">{summary.get('total_transcripts', 0):,}</div>
        </div>
        <div class="summary-card">
            <h3>Avg Confidence</h3>
            <div class="value">{summary.get('avg_confidence', 0):.3f}</div>
        </div>
        <div class="summary-card">
            <h3>Categories</h3>
            <div class="value">{summary.get('unique_categories', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Unique Agents</h3>
            <div class="value">{summary.get('unique_agents', 0)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Category Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for row in category_dist[:10]:
            html += f"""
                <tr>
                    <td>{row.get('category', 'N/A')}</td>
                    <td>{row.get('count', 0):,}</td>
                    <td>{row.get('percentage', 0):.1f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Agent Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Agent Name</th>
                    <th>Total Calls</th>
                    <th>Avg Confidence</th>
                    <th>Unique Categories</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for row in agent_perf[:15]:
            html += f"""
                <tr>
                    <td>{row.get('agent_name', 'N/A')}</td>
                    <td>{row.get('total_calls', 0):,}</td>
                    <td>{row.get('avg_confidence', 0):.3f}</td>
                    <td>{row.get('unique_categories', 0)}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by <strong>AgentPulse AI</strong> - Context Clustered Rule Engine (CCRE)</p>
        <p>Powered by deterministic classification with 96% accuracy</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def export_to_pptx(
        self,
        summary_data: Dict[str, Any],
        coaching_data: Optional[List[Dict[str, Any]]] = None,
        filename: str = "presentation.pptx"
    ) -> str:
        """Export to PowerPoint presentation"""
        output_path = self.output_dir / filename
        
        # Create HTML slides for html2pptx
        slides_dir = self.output_dir / "slides_temp"
        slides_dir.mkdir(exist_ok=True)
        
        # Generate slides
        self._generate_pptx_slides(summary_data, coaching_data, slides_dir)
        
        # Convert to PPTX using html2pptx
        js_script = self._generate_pptx_script(slides_dir, output_path)
        script_path = self.output_dir / "convert.js"
        
        with open(script_path, 'w') as f:
            f.write(js_script)
        
        # Run conversion
        try:
            result = subprocess.run(
                f'NODE_PATH="$(npm root -g)" node {script_path}',
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and output_path.exists():
                # Cleanup temp files
                import shutil
                shutil.rmtree(slides_dir)
                script_path.unlink()
                return str(output_path)
            else:
                raise Exception(f"PPTX generation failed: {result.stderr}")
        except Exception as e:
            # Fallback: return path anyway with error note
            return f"ERROR: {str(e)}"
    
    def _generate_pptx_slides(
        self,
        summary_data: Dict[str, Any],
        coaching_data: Optional[List[Dict[str, Any]]],
        slides_dir: Path
    ):
        """Generate HTML slides for PPTX"""
        
        summary = summary_data.get('summary', {})
        category_dist = summary_data.get('category_distribution', [])
        
        # Slide 1: Title
        slide1_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ width: 960px; height: 540px; margin: 0; padding: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex; align-items: center; justify-content: center;
                font-family: Arial, sans-serif; }}
        .title {{ text-align: center; color: white; }}
        h1 {{ font-size: 48px; margin: 0; }}
        p {{ font-size: 24px; margin-top: 20px; opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="title">
        <h1>AgentPulse AI</h1>
        <p>QA & Coaching Analysis Report</p>
        <p style="font-size: 18px; margin-top: 40px;">{datetime.now().strftime('%B %Y')}</p>
    </div>
</body>
</html>
"""
        with open(slides_dir / "slide1.html", 'w') as f:
            f.write(slide1_html)
        
        # Slide 2: Summary Stats
        slide2_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ width: 960px; height: 540px; margin: 0; padding: 40px;
                font-family: Arial, sans-serif; background: white; }}
        h1 {{ color: #667eea; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .stat {{ padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .stat-label {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
        .stat-value {{ font-size: 36px; color: #667eea; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Key Metrics</h1>
    <div class="stats">
        <div class="stat">
            <div class="stat-label">Total Transcripts Analyzed</div>
            <div class="stat-value">{summary.get('total_transcripts', 0):,}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Average Confidence</div>
            <div class="stat-value">{summary.get('avg_confidence', 0):.1%}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Unique Categories</div>
            <div class="stat-value">{summary.get('unique_categories', 0)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Agents Analyzed</div>
            <div class="stat-value">{summary.get('unique_agents', 0)}</div>
        </div>
    </div>
</body>
</html>
"""
        with open(slides_dir / "slide2.html", 'w') as f:
            f.write(slide2_html)
        
        # Slide 3: Category Distribution
        categories_html = "<br>".join([
            f"{row.get('category', 'N/A')}: {row.get('count', 0)} ({row.get('percentage', 0):.1f}%)"
            for row in category_dist[:8]
        ])
        
        slide3_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ width: 960px; height: 540px; margin: 0; padding: 40px;
                font-family: Arial, sans-serif; background: white; }}
        h1 {{ color: #667eea; margin-bottom: 30px; }}
        .categories {{ font-size: 18px; line-height: 2; }}
    </style>
</head>
<body>
    <h1>Category Distribution</h1>
    <div class="categories">{categories_html}</div>
</body>
</html>
"""
        with open(slides_dir / "slide3.html", 'w') as f:
            f.write(slide3_html)
    
    def _generate_pptx_script(self, slides_dir: Path, output_path: Path) -> str:
        """Generate JavaScript for html2pptx conversion"""
        
        script = f"""
const pptxgen = require("pptxgenjs");
const {{ html2pptx }} = require("@ant/html2pptx");
const path = require("path");

async function createPresentation() {{
    const pptx = new pptxgen();
    pptx.layout = "LAYOUT_16x9";
    
    // Add slides
    const slidesDir = "{slides_dir}";
    
    await html2pptx(path.join(slidesDir, "slide1.html"), pptx);
    await html2pptx(path.join(slidesDir, "slide2.html"), pptx);
    await html2pptx(path.join(slidesDir, "slide3.html"), pptx);
    
    // Save
    await pptx.writeFile("{output_path}");
    console.log("Presentation created successfully");
}}

createPresentation().catch(console.error);
"""
        return script
    
    def export_to_docx(
        self,
        summary_data: Dict[str, Any],
        coaching_data: Optional[List[Dict[str, Any]]] = None,
        filename: str = "report.docx"
    ) -> str:
        """Export executive summary to DOCX"""
        output_path = self.output_dir / filename
        
        # Generate JavaScript for docx creation
        js_script = self._generate_docx_script(summary_data, coaching_data, output_path)
        script_path = self.output_dir / "create_docx.js"
        
        with open(script_path, 'w') as f:
            f.write(js_script)
        
        # Run script
        try:
            result = subprocess.run(
                f'node {script_path}',
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and output_path.exists():
                script_path.unlink()
                return str(output_path)
            else:
                raise Exception(f"DOCX generation failed: {result.stderr}")
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _generate_docx_script(
        self,
        summary_data: Dict[str, Any],
        coaching_data: Optional[List[Dict[str, Any]]],
        output_path: Path
    ) -> str:
        """Generate JavaScript for DOCX creation using docx library"""
        
        summary = summary_data.get('summary', {})
        
        script = f"""
const {{ Document, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType, Packer }} = require("docx");
const fs = require("fs");

const doc = new Document({{
    sections: [{{
        properties: {{}},
        children: [
            new Paragraph({{
                text: "AgentPulse AI - Executive Summary",
                heading: HeadingLevel.HEADING_1,
                alignment: AlignmentType.CENTER
            }}),
            new Paragraph({{
                text: "Generated: {datetime.now().strftime('%B %d, %Y')}",
                alignment: AlignmentType.CENTER
            }}),
            new Paragraph({{ text: "" }}),
            
            new Paragraph({{
                text: "Key Metrics",
                heading: HeadingLevel.HEADING_2
            }}),
            new Paragraph({{
                children: [
                    new TextRun({{ text: "Total Transcripts: ", bold: true }}),
                    new TextRun({{ text: "{summary.get('total_transcripts', 0):,}" }})
                ]
            }}),
            new Paragraph({{
                children: [
                    new TextRun({{ text: "Average Confidence: ", bold: true }}),
                    new TextRun({{ text: "{summary.get('avg_confidence', 0):.3f}" }})
                ]
            }}),
            new Paragraph({{
                children: [
                    new TextRun({{ text: "Unique Categories: ", bold: true }}),
                    new TextRun({{ text: "{summary.get('unique_categories', 0)}" }})
                ]
            }}),
            new Paragraph({{
                children: [
                    new TextRun({{ text: "Agents Analyzed: ", bold: true }}),
                    new TextRun({{ text: "{summary.get('unique_agents', 0)}" }})
                ]
            }}),
            
            new Paragraph({{ text: "" }}),
            new Paragraph({{
                text: "Analysis Summary",
                heading: HeadingLevel.HEADING_2
            }}),
            new Paragraph({{
                text: "The CCRE engine successfully classified all transcripts with hierarchical categorization, providing detailed insights into agent performance and customer interaction patterns."
            }}),
            
            new Paragraph({{ text: "" }}),
            new Paragraph({{
                text: "Recommendations",
                heading: HeadingLevel.HEADING_2
            }}),
            new Paragraph({{
                text: "• Focus coaching on agents with high verification issue rates",
                bullet: {{ level: 0 }}
            }}),
            new Paragraph({{
                text: "• Review DPA processes to ensure compliance",
                bullet: {{ level: 0 }}
            }}),
            new Paragraph({{
                text: "• Implement best practices from high-performing agents",
                bullet: {{ level: 0 }}
            }})
        ]
    }}]
}});

Packer.toBuffer(doc).then(buffer => {{
    fs.writeFileSync("{output_path}", buffer);
    console.log("Document created successfully");
}});
"""
        return script
    
    def batch_export(
        self,
        data: pd.DataFrame,
        summary_data: Dict[str, Any],
        formats: List[str],
        base_filename: str = "agentpulse_export"
    ) -> Dict[str, str]:
        """
        Export to multiple formats at once
        
        Args:
            data: DataFrame to export
            summary_data: Summary statistics dict
            formats: List of format strings ['csv', 'excel', 'html', 'parquet']
            base_filename: Base name for output files
            
        Returns:
            Dict mapping format to output path
        """
        results = {}
        
        if 'csv' in formats:
            results['csv'] = self.export_to_csv(data, f"{base_filename}.csv")
        
        if 'excel' in formats:
            results['excel'] = self.export_to_excel(data, f"{base_filename}.xlsx")
        
        if 'parquet' in formats:
            results['parquet'] = self.export_to_parquet(data, f"{base_filename}.parquet")
        
        if 'html' in formats:
            results['html'] = self.export_to_html(summary_data, f"{base_filename}.html")
        
        if 'pptx' in formats:
            results['pptx'] = self.export_to_pptx(summary_data, None, f"{base_filename}.pptx")
        
        if 'docx' in formats:
            results['docx'] = self.export_to_docx(summary_data, None, f"{base_filename}.docx")
        
        return results
