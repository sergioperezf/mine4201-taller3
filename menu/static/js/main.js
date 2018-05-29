var waitingDialog = waitingDialog || (function ($) {
//Esta funcion carga el modal que dice "cargando" mientras corre Scrapy por detras
	'use strict';
	var $dialog = $('<div class="modal fade" data-backdrop="static" data-keyboard="false" tabindex="-1" role="dialog" aria-hidden="true" style="padding-top:15%; overflow-y:visible;">' +
		'<div class="modal-dialog modal-m">' +
			'<div class="modal-content">' +
				'<div class="modal-header"><h3 style="margin:0;"></h3>' +
				'</div>' +
				'<div class="modal-body">' +
					'<div class="progress progress-striped active" style="margin-bottom:0;"><div class="progress-bar" style="width: 100%">' +
					'</div>' +
				'</div>' +
			'</div>' +
		'</div>' +
	'</div>');

	return {
		show: function (mensaje) {
			$dialog.find('.modal-dialog').attr('class', 'modal-dialog').addClass('modal-m');
			$dialog.find('.progress-bar').attr('class', 'progress-bar');
			$dialog.find('.progress-bar').addClass('progress-bar-striped').addClass('progress-bar-animated');
			$dialog.find('h3').text(mensaje);
			$dialog.modal();
		},
		hide: function () {
			$dialog.modal('hide');
		}
	};

})(jQuery);

function crear_arbol(arbol) {
//Funcion que usa la libreria "Bootstrap Treeview" para visualizar el documento JSON de scrapy
	var $searchableTree = $('#treeview-searchable').treeview({
		data: arbol
	});
	var search = function(e) {
		var pattern = $('#input-search').val();
        	var options = {
			ignoreCase: true,
            		exactMatch: false,
            		revealResults: true
        	};
        	var results = $searchableTree.treeview('search', [ pattern, options ]);
        	$('#search-output').html(output);
    	}
	$('#btn-search').on('click', search);
	$('#input-search').on('keyup', search);
	$('#btn-clear-search').on('click', function (e) {
		$searchableTree.treeview('clearSearch');
		$('#input-search').val('');
		$('#search-output').html('');
	});
	var initSelectableTree = function() {
		return $('#treeview-selectable').treeview({
			data: arbol,
			multiSelect: true
	  });
	};
	var $selectableTree = initSelectableTree();
	var findAllNodes = function() {
		return $selectableTree.treeview('search', [ '\\w+', { ignoreCase: true, exactMatch: false } ]);
	};
	var findSelectableNodes = function() {
		return $selectableTree.treeview('search', [ $('#input-select-node').val(), { ignoreCase: true, exactMatch: false } ]);
	};
	var allNodes = findAllNodes();
	var selectableNodes = findSelectableNodes();
	$('#btn-select-node.select-node').on('click', function (e) {
		selectableNodes = findSelectableNodes();
		$('.select-node').prop('disabled', !(selectableNodes.length >= 1));
		$selectableTree.treeview('unselectNode', [ allNodes, { silent: $('#chk-select-silent').is(':checked') }]);
		$selectableTree.treeview('selectNode', [ selectableNodes, { silent: $('#chk-select-silent').is(':checked') }]);
	});
	$('#btn-unselect-node.select-node').on('click', function (e) {
		$selectableTree.treeview('unselectNode', [ allNodes, { silent: $('#chk-select-silent').is(':checked') }]);
		$selectableTree.treeview('collapseAll');
	});
	$selectableTree.treeview('collapseAll');
}

function anidar_p2(leaf, root) {
//Funcion complementaria a "anidar" para hacerla mas legible
	retorno = []
	for(y in root) {
		retorno.push(root[y]);
		for(x in leaf) {
			if(root[y]['origen'].join('/') == '0' && leaf[x]['origen'].slice(-1).join('/') == root[y].orden) {
				if(typeof retorno[retorno.length - 1]['nodes'] === 'undefined') {
					retorno[retorno.length - 1]['nodes'] = [];
				}
				retorno[retorno.length - 1]['nodes'].push(leaf[x])
			}
			else if(root[y]['origen'].join('/') == leaf[x]['origen'].slice(0, -1).join('/') && leaf[x]['origen'].slice(-1).join('/') == root[y].orden) {
				if(typeof retorno[retorno.length - 1]['nodes'] === 'undefined') {
					retorno[retorno.length - 1]['nodes'] = [];
				}
				retorno[retorno.length - 1]['nodes'].push(leaf[x])
			}
		}
	}
	return retorno;
}

function anidar(arbol) {
//Anido el documento JSON con base en las llaves que se usaron para extraerlo. e.g.: Si traigo el elemento x de la pagina y, la funcion "anidar" sabe que y es hijo de x
	arbol_leaf = [];
	arbol_root = [];
	max_llave = 0;
	num_veces = 0;
	$.each(arbol, function(i, val) {
		if(val[4] > max_llave) { max_llave = val[4]; }
	});
	for(var i = max_llave; i >= 2; i--) {
		arbol_root = [];
		for(var x in arbol) {
			if(arbol[x][4] == i && num_veces == 0) {
				arbol_leaf.push({
					text: x,
					orden: arbol[x][2],
					origen: arbol[x][3].toString().split("-")
				});
			}
			if(arbol[x][4] == i - 1) {
				arbol_root.push({
					text: x,
					orden: arbol[x][2],
					origen: arbol[x][3].toString().split("-")
				});
			}
		}
		num_veces = 1;
		arbol_leaf = anidar_p2(arbol_leaf, arbol_root);
	}
	$.each(arbol_leaf, function(i, val) {
		if(arbol_leaf[i]['text'] == 'Directivos') {
			$.each(arbol_leaf[i]['nodes'], function(i_2, val_2) {
				arbol_leaf[i]['nodes'][i_2]['text'] = '<b>' + arbol_leaf[i]['nodes'][i_2]['text'].substr(0, arbol_leaf[i]['nodes'][i_2]['text'].indexOf(':')) + '</b>' + arbol_leaf[i]['nodes'][i_2]['text'].substr(arbol_leaf[i]['nodes'][i_2]['text'].indexOf(':'));
			});
		}
	});
	var i = arbol_leaf.length;
	while(i--) {
		if(arbol_leaf[i]['text'] == 'Directivos') {
			var j = arbol_leaf[i]['nodes'].length;
			while(j--) {
				if(arbol_leaf[i]['nodes'][j]['text'].indexOf(':') == -1) {
					arbol_leaf[i]['nodes'].splice(j, 1);
				}
			}
		}
	}
//Parte dos de la funcion "anidar". Esta enlaza por el nombre. e.g., si tengo la facultad x, y tengo el directivo (decano) de x, puedo enlazarlo como hijo
	$.each(arbol_leaf, function(i, val) {
		if(arbol_leaf[i]['text'] == 'Directivos') {
			$.each(arbol_leaf[i]['nodes'], function(i_2, val_2) {
				$.each(arbol_leaf, function(i_3, val_3) {
					if(arbol_leaf[i_3]['text'].indexOf('Listado de') != -1) {
						$.each(arbol_leaf[i_3]['nodes'], function(i_4, val_4) {
							if(arbol_leaf[i_3]['nodes'][i_4]['text'].toUpperCase().indexOf(arbol_leaf[i]['nodes'][i_2]['text'].substr(3, arbol_leaf[i]['nodes'][i_2]['text'].indexOf(':') - 7).toUpperCase()) != -1) {
								if(typeof arbol_leaf[i_3]['nodes'][i_4]['nodes'] === 'undefined') {
									arbol_leaf[i_3]['nodes'][i_4]['nodes'] = [];
								}
								arbol_leaf[i_3]['nodes'][i_4]['nodes'].push({
									text: '<b>DIRECTIVO</b>: ' + arbol_leaf[i]['nodes'][i_2]['text']
								});
							}
						});
					}
				});
			});
		}
	});
	crear_arbol(arbol_leaf);
	setTimeout(function() {waitingDialog.hide();}, 2000);
	$('#btn-select-node').prop('disabled', false);
	$('#btn-unselect-node').prop('disabled', false);
}

function traer_facultades_ajax() {
//Llamo al AJAX que me trae un documento JSON desde Scrapy para ponerlo en la liberia "BOOTSTRAP - TREEVIEW"
	$.ajax({
		url : "taller_1/traer_facultades",
		type : "POST",
		data : { valor: 'TRAER_FACULTADES' },
		success : function(ret) {
			anidar(ret);
		},
		error : function(xhr,errmsg,err) {
			setTimeout(function() {waitingDialog.hide();}, 1000);
			alert(xhr.status + ": " + xhr.responseText);
		}
	});
}

jQuery(document).ready(function() {
//Cargo la funcionalidad del boton principal
	$('#traer_facultades').on('click', function(event){
		event.preventDefault();
		waitingDialog.show('Procesando crawler...');
		traer_facultades_ajax();
	});
});
