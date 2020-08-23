from typing import Dict, List, Optional, Union


class Vocab:
    """Vocab class"""

    def __init__(
        self,
        list_of_tokens: List[str] = None,
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<unk>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
        cls_token: Optional[str] = "<cls>",
        sep_token: Optional[str] = "<sep>",
        mask_token: Optional[str] = "<mask>",
        reserved_tokens: Optional[List[str]] = None,
        token_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Instantiating Vocab class
        Args:
            list_of_tokens (List[str]): list of tokens is source of vocabulary. each token is not duplicate
            pad_token (str): the representation for padding token
            unk_token (str): the representation for any unknown token
            bos_token (str): the representation for the special token of beginning-of-sequence token
            eos_token (str): the representation for the special token of end-of-sequence token
            reserved_tokens (List[str]): a list specifying additional tokens to be added to the vocabulary
            token_to_idx (Dict[str, int]): If not `None`, specifies the indices of tokens to be used by the vocabulary.
                                           Each token in `token_to_index` must be part of the Vocab and each index can
                                           only be associated with a single token. `token_to_idx` is not required to
                                           contain a mapping for all tokens. For example, it is valid to only set the
                                            `unknown_token` index to 10 (instead of the default of 0) with
                                           `token_to_idx = {'<unk>': 10}`, assuming that there are at least 10 tokens in
                                            the vocabulary.
        """
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._cls_token = cls_token
        self._sep_token = sep_token
        self._mask_token = mask_token
        self._reserved_tokens = reserved_tokens
        self._special_tokens = []

        for tkn in [
            self._unk_token,
            self._pad_token,
            self._bos_token,
            self._eos_token,
            self._cls_token,
            self._sep_token,
            self._mask_token,
        ]:
            if tkn:
                self._special_tokens.append(tkn)

        if self._reserved_tokens:
            self._special_tokens.extend(self._reserved_tokens)

        if list_of_tokens:
            self._special_tokens.extend(
                list(
                    filter(lambda elm: elm not in self._special_tokens, list_of_tokens)
                )
            )

        self._token_to_idx, self._idx_to_token = self._build(self._special_tokens)

        if token_to_idx:
            self._sort_index_according_to_user_specification(token_to_idx)

        self._embedding = None

    def to_indices(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Looks up indices of text tokens according to the vocabulary
        Args:
            tokens (Union[str, List[str]]): a source token or tokens to be converted
        Returns:
            Union[int, List[int]]: a token index or a list of token indices according to the vocabulary
        """
        if isinstance(tokens, list):
            return [
                self._token_to_idx[tkn]
                if tkn in self._token_to_idx
                else self._token_to_idx[self._unk_token]
                for tkn in tokens
            ]
        else:
            return (
                self._token_to_idx[tokens]
                if tokens in self._token_to_idx
                else self._token_to_idx[self._unk_token]
            )

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts token indices to tokens according to the vocabulary
        Args:
            indices (Union[int, List[int]]): a source token index or token indices to be converted
        Returns:
            Union[str, List[str]]: a token or a list of tokens according to the vocabulary
        """
        if isinstance(indices, list):
            return [self._idx_to_token[idx] for idx in indices]
        else:
            return self._idx_to_token[indices]

    def _build(self, list_of_tokens):
        token_to_idx = {tkn: idx for idx, tkn in enumerate(list_of_tokens)}
        idx_to_token = list_of_tokens
        return token_to_idx, idx_to_token

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self._token_to_idx.keys()):
            raise ValueError(
                "User-specified token_to_idx mapping can only contain "
                "tokens that will be part of the vocabulary."
            )
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError("User-specified indices must not contain duplicates.")
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
            self._token_to_idx
        ):
            raise ValueError(
                "User-specified indices must not be < 0 or >= the number of tokens "
                "that will be in the vocabulary. The current vocab contains {}"
                "tokens.".format(len(self._token_to_idx))
            )

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self._token_to_idx[token]
            ousted_token = self._idx_to_token[new_idx]

            self._token_to_idx[token] = new_idx
            self._token_to_idx[ousted_token] = old_idx
            self._idx_to_token[old_idx] = ousted_token
            self._idx_to_token[new_idx] = token

    def __len__(self):
        return len(self._token_to_idx)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def unk_token(self):
        return self._unk_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def cls_token(self):
        return self._cls_token

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def mask_token(self):
        return self._mask_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array
