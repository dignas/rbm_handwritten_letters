from os import path


def label_by_fname(fname):
	is_capital_str, letter_id_str, _, _ = path.basename(fname).split('.')[0].split('_')

	is_capital = 0
	if is_capital_str == '01':
		is_capital = 1

	letter_id = int(letter_id_str)

	return __number_of_letters * is_capital + letter_id


__number_of_letters = 33
