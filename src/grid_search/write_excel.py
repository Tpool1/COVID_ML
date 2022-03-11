from xlwt import Workbook

class write_excel:
    def __init__(self, sheet_path, hyp_dict_list):
        self.path = sheet_path
        self.hyps = hyp_dict_list

    def run(self):
        wb = Workbook()

        sheet = wb.add_sheet('Sheet 1')

        # add column labels to sheet
        i = 0
        print("hyps", self.hyps)
        perf_tuple = self.hyps[0]
        hyp_dict = perf_tuple[0]
        labels = list(hyp_dict.keys())
        for label in labels:
            sheet.write(0, i, label)

            i = i + 1

        sheet.write(0, i, 'accuracy')

        # add corresponding hyperparameters and performance data to each column
        i = 1
        for perf_tuple in self.hyps:
            hyp_dict = perf_tuple[0]
            accuracy = perf_tuple[1][0]
            f1 = perf_tuple[1][1]
            f1_p = perf_tuple[1][2]
            f1_r = perf_tuple[1][3]
            j = 0
            for label in labels:
                sheet.write(i, j, hyp_dict[label])

                j = j + 1

            sheet.write(i, j, accuracy)
            sheet.write(i+1, j, f1)
            sheet.write(i+2, j, f1_p)
            sheet.write(i+3, j, f1_r)

            # +4 to avoid overwriting 
            i = i + 4

        wb.save(self.path)
