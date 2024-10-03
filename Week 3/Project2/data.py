"""data.py
Reads CSV files, stores data, access/filter data by variable name
Francis O'Hara
CS 251/2: Data Analysis and Visualization
Fall 2024
"""
import numpy as np

class Data:
    """Represents data read in from .csv files
    """
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, cats2levels=None):
        """Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        cats2levels: Python dictionary or None.
                Maps each categorical variable header (var str name) to a list of the unique levels (strings)
                Example:

                For a CSV file that looks like:

                letter,number,greeting
                categorical,categorical,categorical
                a,1,hi
                b,2,hi
                c,2,hi

                cats2levels looks like (key -> value):
                'letter' -> ['a', 'b', 'c']
                'number' -> ['1', '2']
                'greeting' -> ['hi']

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - cats2levels
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        """
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        self.cats2levels = cats2levels

        if filepath is not None:
            self.read(filepath)

    def read(self, filepath):
        """Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called `self.data` at the end
        (think of this as a 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if there should be nothing returned

        TODO:
        1. Set or update your `filepath` instance variable based on the parameter value.
        2. Open and read in the .csv file `filepath` to set `self.data`.
        Parse the file to ONLY store numeric and categorical columns of data in a 2D tabular format (ignore all other
        potential variable types).
            - Numeric data: Store all values as floats.
            - Categorical data: Store values as ints in your list of lists (self.data). Maintain the mapping between the
            int-based and string-based coding of categorical levels in the self.cats2levels dictionary.
        All numeric and categorical values should be added to the SAME list of lists (self.data).
        3. Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        4. Be sure to set the fields: `self.headers`, `self.data`, `self.header2col`, `self.cats2levels`.
        5. Add support for missing data. This arises with there is no entry in a CSV file between adjacent commas.
            For example:
                    letter,number,greeting
                    categorical,categorical,categorical
                     a,1,hi
                     b,,hi
                     c,,hi
            contains two missing values, in the 4th and 5th rows of the 2nd column.
            Handle this differently depending on whether the missing value belongs to a numeric or categorical variable.
            In both cases, you should subsitute a single constant value for the current value to your list of lists (self.data):
            - Numeric data: Subsitute np.nan for the missing value.
            (nan stands for "not a number" — this is a special constant value provided by Numpy).
            - Categorical data: Add a categorical level called 'Missing' to the list of levels in self.cats2levels
            associated with the current categorical variable that has the missing value. Now proceed as if the level
            'Missing' actually appeared in the CSV file and make the current entry in your data list of lists (self.data)
            the INT representing the index (position) of 'Missing' in the level list.
            For example, in the above CSV file example, self.data should look like:
                [[0, 0, 0],
                 [1, 1, 0],
                 [2, 1, 0]]
            and self.cats2levels would look like:
                self.cats2levels['letter'] -> ['a', 'b', 'c']
                self.cats2levels['number'] -> ['1', 'Missing']
                self.cats2levels['greeting'] -> ['hi']

        NOTE:
        - In any CS251 project, you are welcome to create as many helper methods as you'd like. The crucial thing is to
        make sure that the provided method signatures work as advertised.
        - You should only use the basic Python to do your parsing. (i.e. no Numpy or other imports).
        Points will be taken off otherwise.
        - Have one of the CSV files provided on the project website open in a text editor as you code and debug.
        - Run the provided test scripts regularly to see desired outputs and to check your code.
        - It might be helpful to implement support for only numeric data first, test it, then add support for categorical
        variable types afterward.
        - Make use of code from Lab 1a!
        """
        self.filepath = filepath
        self.data = []
        self.headers = []
        self.cats2levels = {}
        self.header2col = {}

        lines = open(filepath, "r").read().split("\n")

        valid_positions = []
        categorical_positions = []
        second_line = lines[1].strip().split(",")

        for i in range(len(second_line)):
            second_line[i] = second_line[i].strip()
            if second_line[i] in ["numeric", "categorical"]:
                valid_positions.append(i)

                if second_line[i] == "categorical":
                    categorical_positions.append(i)

            elif second_line[i] not in ["string", "date"]:
                raise Exception(f"Illegal argument found on line 2: `{second_line[i]}`. Valid arguments for line 2 are `categorical`, `numeric`, `string`, and `date`")


        header_line = lines[0].strip().split(",")
        col2header = {}
        header2col_index = 0

        for i in valid_positions:
            header_line[i] = header_line[i].strip()
            self.headers.append(header_line[i])
            col2header[i] = header_line[i]
            self.header2col[header_line[i]] = header2col_index
            header2col_index += 1

            if i in categorical_positions:
                self.cats2levels[header_line[i]] = []

        for line in lines[2:]:
            if len(line) == 0:
                continue

            line = line.strip().split(",")
            encoded_line = []

            for i in valid_positions:
                line[i] = line[i].strip()
                if i in categorical_positions:
                    if line[i] == "":
                        line[i] = "Missing"
                    if line[i] not in self.cats2levels[col2header[i]]:
                        self.cats2levels[col2header[i]].append(line[i])
                    encoded_line.append(self.cats2levels[col2header[i]].index(line[i]))
                else:
                    if line[i] == "":
                        encoded_line.append(np.nan)
                    else:
                        encoded_line.append(float(line[i]))

            self.data.append(encoded_line)

        self.data = np.array(self.data)

    def __str__(self):
        """toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
        what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.

        NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
        Printing out the categorical variables with string levels would be a small extension.
        """
        result = ""
        # max_header_len = max([len(header) for header in self.headers])
        header_string = "".join(["  " + str(header) + "  " for header in self.headers])
        result += header_string + "\n"
        result += ("-" * len(header_string)) + "\n"

        max_col_len = {}

        for col in range(self.data.shape[1]):
            column = self.data[:, col]
            max_col_len[col] = max([len(str(value)) for value in column])

        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                result += "".join([" " + str(self.data[row, col]).ljust(max_col_len[col]) + " "])
            result += "\n"

        return result


    def get_headers(self):
        """Get list of header names (all variables)

        Returns:
        -----------
        Python list of str.
        """
        return self.headers

    def get_mappings(self):
        """Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        """
        return self.header2col

    def get_cat_level_mappings(self):
        """Get method for mapping between categorical variable names and a list of the respective unique level strings.

        Returns:
        -----------
        Python dictionary. str -> list of str
        """
        return self.cats2levels

    def get_num_dims(self):
        """Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        """
        return len(self.headers)

    def get_num_samples(self):
        """Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        """
        return len(self.data)

    def get_sample(self, rowInd):
        """Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        """
        return self.data[rowInd]

    def get_header_indices(self, headers):
        """Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers` list.
        """
        header_indices = []
        for header in headers:
            header_indices.append(self.header2col[header])
        return header_indices

    def get_all_data(self):
        """Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself. This can be accomplished with numpy's copy
            function.
        """
        return self.data.copy()

    def head(self):
        """Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        """
        return self.data[:5]

    def tail(self):
        """Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        """
        return self.data[-5:]

    def limit_samples(self, start_row, end_row):
        """Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        """
        self.data = self.data[start_row:end_row]

    def select_data(self, headers, rows=[]):
        """Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return column #2 of self.data.
        If rows is not [] (say =[0, 2, 5]), then we do the same thing, but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select. Empty list [] means take all rows.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        """

        cols = list(sorted(self.get_header_indices(headers)))
        rows = list(sorted(rows))
        if len(rows) == 0:
            return self.data[:, cols]
        else:
            return self.data[np.ix_(rows, cols)]

