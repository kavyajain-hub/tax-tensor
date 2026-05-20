import pandas as pd
import numpy as np

class ITCAuditor:
    def __init__(self):
        # Define the exact column mappings we expect to standardize everything
        self.std_columns = {
            "GSTIN": "gstin",
            "Invoice Number": "invoice_no",
            "Tax Amount": "tax_amount"
        }

    def _normalize_invoice(self, inv_series):
        """
        Strips all non-alphanumeric characters and converts to uppercase 
        to ensure 'INV/001' matches 'INV-001' or 'inv001'.
        """
        return inv_series.astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.upper()

    def reconcile(self, internal_df: pd.DataFrame, gstr2b_df: pd.DataFrame) -> pd.DataFrame:
        """
        Core reconciliation engine comparing internal registers with government data.
        """
        # 1. Standardize column names for processing (assuming basic mapping for now)
        int_df = internal_df.rename(columns=lambda x: x.strip().title())
        g2b_df = gstr2b_df.rename(columns=lambda x: x.strip().title())

        # 2. Create the normalization key for robust joining
        int_df['match_key'] = self._normalize_invoice(int_df['Invoice Number'])
        g2b_df['match_key'] = self._normalize_invoice(g2b_df['Invoice Number'])

        # 3. Perform the outer merge to find gaps
        # indicator=True adds a '_merge' column telling us where the row came from
        merged = pd.merge(
            int_df, 
            g2b_df, 
            on=['Gstin', 'match_key'], 
            how='outer', 
            suffixes=('_internal', '_gstr2b'),
            indicator=True
        )

        # 4. Categorize the discrepancies
        results = []
        for index, row in merged.iterrows():
            if row['_merge'] == 'left_only':
                # In internal books, but missing in GSTR-2B -> HIGH RISK (ITC Loss)
                results.append({
                    "Vendor GSTIN": row['Gstin'],
                    "Original Invoice": row['Invoice Number_internal'],
                    "Discrepancy": "Missing in GSTR-2B",
                    "ITC at Risk": row['Tax Amount_internal'],
                    "Severity": "🔴 HIGH"
                })
            elif row['_merge'] == 'right_only':
                # In GSTR-2B, but missing internally -> Needs review (Unrecorded purchase)
                results.append({
                    "Vendor GSTIN": row['Gstin'],
                    "Original Invoice": row['Invoice Number_gstr2b'],
                    "Discrepancy": "Missing in Internal Books",
                    "ITC at Risk": row['Tax Amount_gstr2b'],
                    "Severity": "🟡 MEDIUM"
                })
            elif row['_merge'] == 'both':
                # Matched! Now check for value discrepancies
                diff = abs(row['Tax Amount_internal'] - row['Tax Amount_gstr2b'])
                # Allow a small tolerance for rounding errors (e.g., ₹ 1.00)
                if diff > 1.0:
                    results.append({
                        "Vendor GSTIN": row['Gstin'],
                        "Original Invoice": row['Invoice Number_internal'],
                        "Discrepancy": "Value Mismatch",
                        "ITC at Risk": diff,
                        "Severity": "🟠 MEDIUM-HIGH"
                    })

        return pd.DataFrame(results)