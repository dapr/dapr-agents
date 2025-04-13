from dapr_agents.document.fetcher.base import FetcherBase
from dapr_agents.types.document import Document
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


class ArxivFetcher(FetcherBase):
    """
    Fetcher for interacting with the arXiv API.
    """

    max_results: int = 10
    include_full_metadata: bool = False

    def search(
        self,
        query: str,
        from_date: Union[str, datetime, None] = None,
        to_date: Union[str, datetime, None] = None,
        download: bool = False,
        dirpath: Path = Path("./"),
        include_summary: bool = False,
        **kwargs,
    ) -> Union[List[Dict], List["Document"]]:
        """
        Search for papers on arXiv and optionally download them.

        Args:
            query (str): The search query.
            from_date (Union[str, datetime, None]): Start date for the search in 'YYYYMMDD' format or as a datetime object.
            to_date (Union[str, datetime, None]): End date for the search in 'YYYYMMDD' format or as a datetime object.
            download (bool): Whether to download the papers as PDFs.
            dirpath (Path): Directory path for the downloads (used if download=True).
            include_summary (bool): Whether to include the paper summary in the returned metadata or documents. Defaults to False.
            **kwargs: Additional search parameters (e.g., sort_by).

        Returns:
            Union[List[Dict], List[Document]]: A list of metadata dictionaries if `download=True`,
            otherwise a list of `Document` objects.

        Examples:
            >>> fetcher = ArxivFetcher()
            >>> fetcher.search("quantum computing")
            # Searches for papers related to "quantum computing".

            >>> fetcher.search("machine learning", from_date="20240101", to_date="20240131")
            # Searches for papers on "machine learning" submitted in January 2024.

            >>> fetcher.search("cybersecurity", from_date=datetime(2024, 1, 1), to_date=datetime(2024, 1, 31))
            # Same as above but using datetime objects for date filtering.

            >>> fetcher.search("artificial intelligence", download=True, dirpath=Path("./downloads"))
            # Searches for papers on "artificial intelligence" and downloads the PDFs to "./downloads".
        """
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "The `arxiv` library is required to use the ArxivFetcher. "
                "Install it with `pip install arxiv`."
            )

        logger.info(f"Searching for query: {query}")

        # Enforce that both from_date and to_date are provided if one is specified
        if (from_date and not to_date) or (to_date and not from_date):
            raise ValueError(
                "Both 'from_date' and 'to_date' must be specified if one is provided."
            )

        # Add date filter if both from_date and to_date are provided
        if from_date and to_date:
            from_date_str = self._format_date(from_date)
            to_date_str = self._format_date(to_date)
            date_filter = f"submittedDate:[{from_date_str} TO {to_date_str}]"
            query = f"{query} AND {date_filter}"

        search = arxiv.Search(
            query=query,
            max_results=kwargs.get("max_results", self.max_results),
            sort_by=kwargs.get("sort_by", arxiv.SortCriterion.SubmittedDate),
            sort_order=kwargs.get("sort_order", arxiv.SortOrder.Descending),
        )
        results = list(search.results())
        logger.info(f"Found {len(results)} results for query: {query}")

        return self._process_results(results, download, dirpath, include_summary)

    def search_by_id(
        self,
        content_id: str,
        download: bool = False,
        dirpath: Path = Path("./"),
        include_summary: bool = False,
    ) -> Union[Optional[Dict], Optional[Document]]:
        """
        Search for a specific paper by its arXiv ID and optionally download it.

        Args:
            content_id (str): The arXiv ID of the paper.
            download (bool): Whether to download the paper.
            dirpath (Path): Directory path for the download (used if download=True).
            include_summary (bool): Whether to include the paper summary in the returned metadata or document. Defaults to False.

        Returns:
            Union[Optional[Dict], Optional[Document]]: Metadata dictionary if `download=True`,
            otherwise a `Document` object.

        Examples:
            >>> fetcher = ArxivFetcher()
            >>> fetcher.search_by_id("1234.5678")
            # Searches for the paper with arXiv ID "1234.5678".

            >>> fetcher.search_by_id("1234.5678", download=True, dirpath=Path("./downloads"))
            # Searches for the paper with arXiv ID "1234.5678" and downloads it to "./downloads".
        """
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "The `arxiv` library is required to use the ArxivFetcher. "
                "Install it with `pip install arxiv`."
            )

        logger.info(f"Searching for paper by ID: {content_id}")
        try:
            search = arxiv.Search(id_list=[content_id])
            result = next(search.results(), None)
            if not result:
                logger.warning(f"No result found for ID: {content_id}")
                return None

            return self._process_results([result], download, dirpath, include_summary)[
                0
            ]
        except Exception as e:
            logger.error(f"Error fetching result for ID {content_id}: {e}")
            return None

    def _process_results(
        self, results: List[Any], download: bool, dirpath: Path, include_summary: bool
    ) -> Union[List[Dict], List["Document"]]:
        """
        Process arXiv search results.

        Args:
            results (List[Any]): The list of arXiv result objects.
            download (bool): Whether to download the papers as PDFs.
            dirpath (Path): Directory path for the downloads (used if download=True).
            include_summary (bool): Whether to include the paper summary in the returned metadata or documents.

        Returns:
            Union[List[Dict], List[Document]]: A list of metadata dictionaries if `download=True`,
            otherwise a list of `Document` objects.
        """
        if download:
            metadata_list = []
            for result in results:
                file_path = self._download_result(result, dirpath)
                metadata_list.append(
                    self._format_result_metadata(
                        result, file_path=file_path, include_summary=include_summary
                    )
                )
            return metadata_list
        else:
            documents = []
            for result in results:
                metadata = self._format_result_metadata(
                    result, include_summary=include_summary
                )
                text = result.summary.strip()
                documents.append(Document(text=text, metadata=metadata))
            return documents

    def _download_result(self, result: Any, dirpath: Path) -> Optional[str]:
        """
        Download a paper from an arXiv result object.

        Args:
            result (Any): The arXiv result object.
            dirpath (Path): Directory path for the download.

        Returns:
            Optional[str]: Path to the downloaded file, or None if the download failed.
        """
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            filename = result._get_default_filename()
            file_path = dirpath / filename
            logger.info(f"Downloading paper to {file_path}")
            result.download_pdf(dirpath=str(dirpath), filename=filename)
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to download paper {result.title}: {e}")
            return None

    def _format_result_metadata(
        self,
        result: Any,
        file_path: Optional[str] = None,
        include_summary: bool = False,
    ) -> Dict:
        """
        Format metadata from an arXiv result, optionally including file path and summary.

        Args:
            result (Any): The arXiv result object.
            file_path (Optional[str]): Path to the downloaded file.
            include_summary (bool): Whether to include the summary in the metadata.

        Returns:
            Dict: A dictionary containing formatted metadata.
        """
        metadata = {
            "entry_id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d"),
            "primary_category": result.primary_category,
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "file_path": file_path,
        }

        if self.include_full_metadata:
            metadata.update(
                {
                    "links": result.links,
                    "authors_comment": result.comment,
                    "DOI": result.doi,
                    "journal_reference": result.journal_ref,
                }
            )

        if include_summary:
            metadata["summary"] = result.summary.strip()

        return {key: value for key, value in metadata.items() if value is not None}

    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        Format a date into the 'YYYYMMDDHHMM' format required by the arXiv API.

        Args:
            date (Union[str, datetime]): The date to format. Can be a string in 'YYYYMMDD' or
                'YYYYMMDDHHMM' format, or a datetime object.

        Returns:
            str: The formatted date string.

        Raises:
            ValueError: If the provided date string is not in the correct format or invalid.

        Examples:
            >>> fetcher = ArxivFetcher()
            >>> fetcher._format_date("20240101")
            '202401010000'

            >>> fetcher._format_date("202401011200")
            '202401011200'

            >>> fetcher._format_date(datetime(2024, 1, 1, 12, 0))
            '202401011200'

            >>> fetcher._format_date("invalid_date")
            # Raises ValueError: Invalid date format: invalid_date. Use 'YYYYMMDD' or 'YYYYMMDDHHMM'.
        """
        if isinstance(date, str):
            # Check if the string matches the basic format
            if not re.fullmatch(r"^\d{8}(\d{4})?$", date):
                raise ValueError(
                    f"Invalid date format: {date}. Use 'YYYYMMDD' or 'YYYYMMDDHHMM'."
                )

            # Validate that it is a real date
            try:
                if len(date) == 8:  # 'YYYYMMDD'
                    datetime.strptime(date, "%Y%m%d")
                elif len(date) == 12:  # 'YYYYMMDDHHMM'
                    datetime.strptime(date, "%Y%m%d%H%M")
            except ValueError as e:
                raise ValueError(f"Invalid date value: {date}. {str(e)}")

            return date
        elif isinstance(date, datetime):
            return date.strftime("%Y%m%d%H%M")
        else:
            raise ValueError(
                "Invalid date input. Provide a string in 'YYYYMMDD', 'YYYYMMDDHHMM' format, or a datetime object."
            )
