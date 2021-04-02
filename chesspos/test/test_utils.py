import chesspos.utils.utils as ut

def test_correct_file_ending():
	assert ut.correct_file_ending("data/hello", "txt") == "data/hello.txt"
	assert ut.correct_file_ending("one_file.pyc", "pyc") == "one_file.pyc"
